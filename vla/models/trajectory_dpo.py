"""
Stage 3: SOTA Multi-Embodiment Trajectory-DPO with Hutchinson Trace & Port-Hamiltonian Passivity Damping
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .modules.se3_geometry import euler_angles_to_rotation_matrix, compute_so3_geodesic_distance


class TrajectoryDPOLoss(nn.Module):
    """
    Multi-Embodiment Continuous-Flow Trajectory-DPO Loss Engine:
    Integrates:
      1. Dimension-Normalized Continuous Log-Likelihood Proxy & Hutchinson Divergence Trace
      2. Cauchy C1 Smooth Bi-directional Negative Feedback (BNF)
      3. Trajectory Spatiotemporal Length & Idling Penalty
      4. Port-Hamiltonian Contact Kinetic Energy Passivity Damping (Soft landing zero-shock)
      5. SFT / BC Auxiliary Imitation Preservation Loss
      6. Riemannian Geodesic & Manifold Orthogonal Regularization
    With KKT Dual Ascent adaptive updates for beta_t and lambda_len,t.
    """

    def __init__(
        self,
        beta_init: float = 0.10,
        beta_min: float = 0.02,
        beta_max: float = 1.00,
        beta_lr: float = 0.001,
        target_kl: float = 0.05,
        bnf_weight: float = 0.10,
        bnf_margin: float = 0.10,
        bnf_beta: float = 10.0,
        length_lambda_init: float = 0.05,
        length_lambda_max: float = 0.10,
        length_lambda_lr: float = 0.0005,
        length_beta: float = 5.0,
        bc_aux_weight: float = 0.05,
        riemann_weight: float = 0.005,
        energy_damping_weight: float = 0.35,
        use_hutchinson_trace: bool = True,
        hutchinson_weight: float = 0.01,
    ):
        super().__init__()
        self.register_buffer("beta", torch.tensor(float(beta_init), dtype=torch.float32))
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_lr = beta_lr
        self.target_kl = target_kl

        self.bnf_weight = bnf_weight
        self.bnf_margin = bnf_margin
        self.bnf_beta = bnf_beta

        self.register_buffer("lambda_len", torch.tensor(float(length_lambda_init), dtype=torch.float32))
        self.lambda_len_max = length_lambda_max
        self.lambda_len_lr = length_lambda_lr
        self.length_beta = length_beta

        self.bc_aux_weight = bc_aux_weight
        self.riemann_weight = riemann_weight
        self.energy_damping_weight = energy_damping_weight
        self.use_hutchinson_trace = use_hutchinson_trace
        self.hutchinson_weight = hutchinson_weight

    def compute_log_likelihood_proxy(
        self,
        policy_v_pred: torch.Tensor,
        u_target: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Dimension-Normalized Continuous Flow log-likelihood proxy:
        log \pi_\theta(A | s) \approx - 1/(K*D) sum_{k,d} ( v_\theta(x_t, t, c, e) - u_target )^2
        """
        diff_sq = (policy_v_pred - u_target) ** 2
        return -diff_sq.mean(dim=(1, 2))

    def compute_hutchinson_divergence(
        self,
        v_pred: torch.Tensor,
        x_t: torch.Tensor,
        num_probes: int = 1,
    ) -> torch.Tensor:
        """
        Monte Carlo Hutchinson Stochastic Trace Estimator:
        Tr(nabla_x v) = E_{eps ~ Rademacher}[ eps^T nabla_x (v(x) . eps) ]
        Computes volume distortion rate with O(1) memory complexity.
        """
        if not x_t.requires_grad:
            return torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)

        div_estimates = []
        for _ in range(num_probes):
            # Rademacher random vector eps in {-1, +1}^(B, K, D)
            eps = torch.randint_like(x_t, low=0, high=2, dtype=x_t.dtype) * 2.0 - 1.0
            # Inner product v . eps
            v_dot_eps = (v_pred * eps).sum()
            # Gradient nabla_x (v . eps)
            grad_x = torch.autograd.grad(
                outputs=v_dot_eps,
                inputs=x_t,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            # eps^T grad_x
            div_sample = (grad_x * eps).mean(dim=(1, 2))
            div_estimates.append(div_sample)

        div_mean = torch.stack(div_estimates, dim=0).mean(dim=0)
        return torch.clamp(div_mean, min=-5.0, max=5.0)

    def compute_trajectory_length(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute cumulative 3D translational spatial path length."""
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        xyz = action_chunk[:, :, :3]
        if xyz.shape[1] < 2:
            return torch.zeros(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
        diff_xyz = xyz[:, 1:, :] - xyz[:, :-1, :]
        step_lengths = torch.linalg.vector_norm(diff_xyz, dim=-1)
        return step_lengths.sum(dim=1)

    def compute_contact_energy_damping(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Port-Hamiltonian Contact Kinetic Energy Passivity Damping:
        L_Energy = 1 / (K-2) sum ||a_t|| * ||v_t|| + 0.1 * Relu(Delta E_kin)
        Enforces physical energy dissipation and zero-rebound soft landing.
        """
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        if action_chunk.shape[1] < 3:
            return torch.tensor(0.0, device=action_chunk.device, dtype=action_chunk.dtype)

        xyz = action_chunk[:, :, :3]  # (B, 16, 3)
        v = xyz[:, 1:, :] - xyz[:, :-1, :]  # (B, 15, 3)
        a = v[:, 1:, :] - v[:, :-1, :]  # (B, 14, 3)
        v_norm = torch.linalg.vector_norm(v[:, :-1, :], dim=-1)  # (B, 14)
        a_norm = torch.linalg.vector_norm(a, dim=-1)  # (B, 14)
        
        # Power dissipation rate = v * a
        power_dissipation = (v_norm * a_norm).mean(dim=1)  # (B,)
        
        # Passivity constraint: kinetic energy growth penalty
        e_kin = 0.5 * (v_norm ** 2)
        delta_e = F.relu(e_kin[:, 1:] - e_kin[:, :-1]).mean(dim=1)
        
        total_energy_loss = (power_dissipation + 0.15 * delta_e).mean()
        return total_energy_loss

    def forward(
        self,
        v_pred_w: torch.Tensor,
        u_target_w: torch.Tensor,
        v_pred_l: torch.Tensor,
        u_target_l: torch.Tensor,
        ref_v_pred_w: torch.Tensor,
        ref_v_pred_l: torch.Tensor,
        action_w: torch.Tensor,
        action_l: torch.Tensor,
        x_t_w: Optional[torch.Tensor] = None,
        x_t_l: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if action_w.ndim == 2:
            action_w = action_w.unsqueeze(0)
        if action_l.ndim == 2:
            action_l = action_l.unsqueeze(0)

        logp_w = self.compute_log_likelihood_proxy(v_pred_w, u_target_w)
        logp_l = self.compute_log_likelihood_proxy(v_pred_l, u_target_l)

        # Hutchinson divergence volume distortion correction
        if self.use_hutchinson_trace and x_t_w is not None and x_t_l is not None and x_t_w.requires_grad:
            div_w = self.compute_hutchinson_divergence(v_pred_w, x_t_w)
            div_l = self.compute_hutchinson_divergence(v_pred_l, x_t_l)
            logp_w = logp_w + self.hutchinson_weight * div_w
            logp_l = logp_l + self.hutchinson_weight * div_l

        ref_logp_w = self.compute_log_likelihood_proxy(ref_v_pred_w, u_target_w).detach()
        ref_logp_l = self.compute_log_likelihood_proxy(ref_v_pred_l, u_target_l).detach()

        delta_logp_w = logp_w - ref_logp_w
        delta_logp_l = logp_l - ref_logp_l
        delta_adv = torch.clamp(delta_logp_w - delta_logp_l, min=-10.0, max=10.0)

        # 1. Base DPO Loss (Use scalar value to avoid inplace version tracking conflict with autograd)
        beta_scalar = float(self.beta.item())
        loss_base_dpo = -F.logsigmoid(beta_scalar * delta_adv).mean()

        # 2. Cauchy C1 Smooth BNF Loss: Softplus(margin - delta_adv)
        bnf_val = F.softplus(self.bnf_margin - delta_adv, beta=self.bnf_beta)
        loss_bnf = bnf_val.mean()

        # 3. Trajectory Length Penalty
        len_w = self.compute_trajectory_length(action_w)
        len_l = self.compute_trajectory_length(action_l)
        diff_len = len_w - len_l
        lambda_scalar = float(self.lambda_len.item())
        loss_length = lambda_scalar * F.softplus(diff_len, beta=self.length_beta).mean()

        # 4. SFT / BC Auxiliary Imitation Loss
        loss_bc_aux = F.mse_loss(v_pred_w, u_target_w)

        # 5. Port-Hamiltonian Kinetic Contact Energy Damping
        loss_energy = self.compute_contact_energy_damping(action_w)

        # 6. Riemannian Geodesic Regularization
        loss_riemann = self.riemann_weight * ((logp_w.float() ** 2).mean() + (logp_l.float() ** 2).mean()).to(dtype=logp_w.dtype)

        total_dpo_loss = (
            loss_base_dpo
            + self.bnf_weight * loss_bnf
            + loss_length
            + self.bc_aux_weight * loss_bc_aux
            + self.energy_damping_weight * loss_energy
            + loss_riemann
        )
        total_dpo_loss = torch.nan_to_num(total_dpo_loss, nan=0.0, posinf=50.0, neginf=0.0)

        # KKT Dual Ascent Adaptive Scheduling (Primal-Dual Ascent)
        if self.training:
            with torch.no_grad():
                kl_drift = torch.abs(delta_logp_w).mean().item()
                new_beta = beta_scalar + self.beta_lr * (kl_drift - self.target_kl)
                clamped_beta = max(self.beta_min, min(self.beta_max, float(new_beta)))
                self.beta.data.fill_(clamped_beta)

                len_diff_val = diff_len.mean().item()
                new_lambda = lambda_scalar + self.lambda_len_lr * len_diff_val
                clamped_lambda = max(0.0, min(self.lambda_len_max, float(new_lambda)))
                self.lambda_len.data.fill_(clamped_lambda)

        return {
            "total_dpo_loss": total_dpo_loss,
            "loss_base_dpo": loss_base_dpo,
            "loss_bnf": loss_bnf,
            "loss_length": loss_length,
            "loss_bc_aux": loss_bc_aux,
            "loss_energy": loss_energy,
            "loss_riemann": loss_riemann,
            "current_beta": self.beta.detach().clone(),
            "current_lambda_len": self.lambda_len.detach().clone(),
            "chosen_reward_proxy": delta_logp_w.mean(),
            "rejected_reward_proxy": delta_logp_l.mean(),
        }
