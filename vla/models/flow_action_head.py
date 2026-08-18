"""
Stage 2: Lie Group SE(3) Conditional Flow Matching (CFM) Action Head with FiLM Embodiment Modulation
Supports: Euler ODE & 2nd-Order Heun (RK2) Numerical Solvers
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .modules.embodiment_encoder import FiLMLayer
from .modules.se3_geometry import (
    euler_angles_to_rotation_matrix,
    compute_so3_geodesic_distance,
    compute_se3_action_geodesic_error,
)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal Positional Embedding for scalar time t in [0, 1]."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t.float() * freqs[None, :]
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embed.to(dtype=t.dtype if torch.is_floating_point(t) else torch.float32)


class FiLMResidualMLPBlock(nn.Module):
    """Residual MLP Block modulated with Embodiment FiLM affine scaling and shifting."""

    def __init__(self, hidden_dim: int, embodiment_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.film = FiLMLayer(condition_dim=embodiment_dim, target_dim=hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, emb_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(self.fc1(x))
        if emb_feat is not None:
            x = self.film(x, emb_feat)
        x = self.norm2(x)
        x = self.fc2(x)
        return residual + x


class FlowActionHead(nn.Module):
    """
    Decoupled SE(3) Optimal Transport Conditional Flow Matching (OT-CFM) Action Generator:
    Generates 7-DoF continuous action chunks A in R^(16 x 7), with Lie Group SO(3) geodesic loss
    and FiLM multi-embodiment adaptation.
    """

    def __init__(
        self,
        context_dim: int = 768,
        chunk_size: int = 16,
        action_dim: int = 7,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        embodiment_dim: int = 128,
        num_layers: int = 4,
        pos_weight: float = 1.0,
        rot_geodesic_weight: float = 1.0,
        gripper_weight: float = 0.5,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.action_flat_dim = chunk_size * action_dim  # 16 * 7 = 112
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.pos_weight = pos_weight
        self.rot_weight = rot_geodesic_weight
        self.gripper_weight = gripper_weight

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.in_proj = nn.Linear(self.action_flat_dim + context_dim, hidden_dim)

        # FiLM-modulated residual blocks
        self.blocks = nn.ModuleList(
            [FiLMResidualMLPBlock(hidden_dim=hidden_dim, embodiment_dim=embodiment_dim) for _ in range(num_layers)]
        )

        # Output velocity head
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.action_flat_dim)

    def forward_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        embodiment_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute predicted velocity field v_θ(x_t, t, c, e).
        """
        B = x_t.shape[0]
        x_flat = x_t.view(B, self.action_flat_dim)
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=5.0, neginf=-5.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=5.0, neginf=-5.0)

        t_emb = self.time_mlp(self.time_embed(t).to(dtype=context.dtype))
        h = self.in_proj(torch.cat([x_flat, context], dim=-1)) + t_emb

        for block in self.blocks:
            h = block(h, embodiment_feat)

        h = self.out_norm(h)
        v_pred_flat = self.out_proj(h)
        v_pred_flat = torch.nan_to_num(v_pred_flat, nan=0.0, posinf=10.0, neginf=-10.0)

        return v_pred_flat.view(B, self.chunk_size, self.action_dim)

    def compute_cfm_loss(
        self,
        action_target: torch.Tensor,
        context: torch.Tensor,
        embodiment_feat: Optional[torch.Tensor] = None,
        x_0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute OT-CFM loss decoupled into:
          - Cartesian Translation velocity loss
          - SO(3) Euler / Lie Algebra angular velocity loss
          - Gripper transition loss
        """
        if not torch.is_floating_point(action_target):
            action_target = action_target.float()
        if action_target.ndim == 2:
            action_target = action_target.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if embodiment_feat is not None and embodiment_feat.ndim == 1:
            embodiment_feat = embodiment_feat.unsqueeze(0)

        B = action_target.shape[0]
        device = action_target.device

        if x_0 is None:
            x_0 = torch.randn_like(action_target)
            x_0 = torch.clamp(x_0, min=-4.0, max=4.0)
        if t is None:
            t = torch.rand(B, device=device)

        t_bc = t.view(B, 1, 1)
        x_1 = torch.clamp(torch.nan_to_num(action_target, nan=0.0), min=-10.0, max=10.0)

        x_t = (1.0 - t_bc) * x_0 + t_bc * x_1
        u_t = x_1 - x_0

        v_pred = self.forward_velocity(x_t, t, context, embodiment_feat)

        # Decouple losses
        v_pos = v_pred[:, :, :3]
        u_pos = u_t[:, :, :3]
        loss_pos = F.mse_loss(v_pos, u_pos)
        loss_pos = torch.nan_to_num(loss_pos, nan=0.0, posinf=20.0, neginf=0.0)

        # Lie Group SO(3) Decoupled Loss: Tangent Space MSE + Ultra-Stable Smooth SO(3) Chordal Distance
        v_rot = v_pred[:, :, 3:6]
        u_rot = u_t[:, :, 3:6]
        loss_rot_tangent = F.mse_loss(v_rot, u_rot)
        
        # Smooth SO(3) Chordal metric: L_chordal = 1/3 * (3 - Tr(R_pred @ R_tgt^T))
        # Everywhere smooth C^infty, strictly convex near identity, zero NaN singularity in FP16/AMP
        R_pred = euler_angles_to_rotation_matrix(v_rot.float())
        R_tgt = euler_angles_to_rotation_matrix(u_rot.float())
        R_rel = torch.matmul(R_pred, R_tgt.transpose(-1, -2))
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        loss_rot_chordal = torch.clamp((3.0 - trace) / 3.0, min=0.0, max=4.0).mean().to(dtype=v_rot.dtype)

        loss_rot = loss_rot_tangent + 0.50 * loss_rot_chordal
        loss_rot = torch.nan_to_num(loss_rot, nan=0.0, posinf=20.0, neginf=0.0)

        v_grip = v_pred[:, :, 6]
        u_grip = u_t[:, :, 6]
        loss_grip = F.mse_loss(v_grip, u_grip)
        loss_grip = torch.nan_to_num(loss_grip, nan=0.0, posinf=20.0, neginf=0.0)

        total_cfm_loss = self.pos_weight * loss_pos + self.rot_weight * loss_rot + self.gripper_weight * loss_grip
        total_cfm_loss = torch.nan_to_num(total_cfm_loss, nan=0.0, posinf=50.0, neginf=0.0)

        info = {
            "loss_cfm": total_cfm_loss,
            "loss_pos": loss_pos,
            "loss_rot": loss_rot,
            "loss_grip": loss_grip,
            "x_0": x_0,
            "x_t": x_t,
            "u_t": u_t,
            "t": t,
            "v_pred": v_pred,
        }
        return total_cfm_loss, info

    @torch.no_grad()
    def sample_actions(
        self,
        context: torch.Tensor,
        embodiment_feat: Optional[torch.Tensor] = None,
        num_steps: int = 6,
        solver: str = "euler",  # "euler" or "heun" (RK2)
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Euler / 2nd-order Heun (RK2) numerical solver for multi-embodiment action generation.
        """
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if embodiment_feat is not None and embodiment_feat.ndim == 1:
            embodiment_feat = embodiment_feat.unsqueeze(0)

        B = context.shape[0]
        device = context.device

        if x_0 is None:
            x = torch.randn((B, self.chunk_size, self.action_dim), device=device)
        else:
            x = x_0.clone()

        num_steps = max(1, int(num_steps))
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_scalar = step * dt
            t_tensor = torch.full((B,), t_scalar, device=device, dtype=torch.float32)
            v1 = self.forward_velocity(x, t_tensor, context, embodiment_feat)

            if solver == "heun" and step < num_steps - 1:
                # 2nd-order Predictor-Corrector Heun step
                x_pred = x + dt * v1
                t_next = torch.full((B,), (step + 1) * dt, device=device, dtype=torch.float32)
                v2 = self.forward_velocity(x_pred, t_next, context, embodiment_feat)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                # Standard 1st-order Euler step
                x = x + dt * v1

        action_pred = torch.clamp(x, -1.0, 1.0)
        return action_pred
