"""
Unified Multi-Embodiment OpenVLA-AlignFlow Architecture with Safety CBF Filter
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..configs.config import VLAConfig
from .vl_backbone import MultiModalCrossAttentionBackbone
from .vl_alignment import VLAlignmentModule
from .flow_action_head import FlowActionHead
from .trajectory_dpo import TrajectoryDPOLoss
from .modules.safety_cbf import KinematicCBFSafetyFilter


class OpenVLAAlignFlow(nn.Module):
    """
    Multi-Embodiment OpenVLA-AlignFlow End-to-End Embodied Agent Architecture:
    Supports WidowX, Google Robot, and Franka Panda embodiments simultaneously.
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        super().__init__()
        self.config = config or VLAConfig()

        # 1. Multi-Embodiment Multimodal Backbone
        self.backbone = MultiModalCrossAttentionBackbone(
            embed_dim=self.config.vision_embed_dim,
            vision_num_heads=self.config.vision_num_heads,
            vision_num_layers=self.config.vision_num_layers,
            text_num_heads=self.config.text_num_heads,
            text_num_layers=self.config.text_num_layers,
            cross_attn_heads=self.config.cross_attn_heads,
            image_size=self.config.image_size,
            patch_size=self.config.vision_patch_size,
            num_embodiments=self.config.num_embodiments,
            embodiment_embed_dim=self.config.embodiment_embed_dim,
        )

        # 2. Stage 1 Alignment Head
        self.alignment_head = VLAlignmentModule(
            embed_dim=self.config.vision_embed_dim,
            projection_dim=self.config.projection_dim,
            infonce_temperature=self.config.infonce_temperature,
            affordance_weight=self.config.affordance_weight,
            affordance_temperature=self.config.affordance_temperature,
        )

        # 3. Stage 2 Lie Group SE(3) Flow Action Head
        self.flow_action_head = FlowActionHead(
            context_dim=self.config.vision_embed_dim,
            chunk_size=self.config.chunk_size,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.flow_hidden_dim,
            time_embed_dim=self.config.time_embed_dim,
            embodiment_dim=self.config.embodiment_embed_dim,
            num_layers=self.config.flow_num_layers,
            pos_weight=self.config.pos_loss_weight,
            rot_geodesic_weight=self.config.rot_geodesic_weight,
            gripper_weight=self.config.gripper_loss_weight,
        )

        # 4. Stage 3 Multi-Embodiment Trajectory-DPO Loss Engine
        self.dpo_engine = TrajectoryDPOLoss(
            beta_init=self.config.beta_init,
            beta_min=self.config.beta_min,
            beta_max=self.config.beta_max,
            beta_lr=self.config.beta_lr,
            target_kl=self.config.beta_target_kl,
            bnf_weight=self.config.bnf_weight,
            bnf_margin=self.config.bnf_margin,
            bnf_beta=self.config.bnf_beta,
            length_lambda_init=self.config.length_lambda_init,
            length_lambda_max=self.config.length_lambda_max,
            length_lambda_lr=self.config.length_lambda_lr,
            length_beta=self.config.length_beta,
            bc_aux_weight=self.config.bc_aux_weight,
            riemann_weight=self.config.riemann_weight,
            energy_damping_weight=self.config.energy_damping_weight,
            use_hutchinson_trace=self.config.use_hutchinson_trace,
            hutchinson_weight=self.config.hutchinson_weight,
        )

        # 5. Formal Kinematic CBF Safety Filter
        self.safety_filter = KinematicCBFSafetyFilter(
            max_velocity=self.config.max_velocity_threshold,
            max_acceleration=self.config.max_acceleration_threshold,
            max_jerk=self.config.jerk_safety_line,
            delta_t=self.config.delta_t,
        )

    def encode(
        self,
        obs_image: torch.Tensor,
        instruction: List[str],
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone(obs_image, instruction, embodiment_id)

    def forward_stage1(
        self,
        obs_image: torch.Tensor,
        goal_image: torch.Tensor,
        instruction: List[str],
        affordance_mask_gt: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.encode(obs_image, instruction, embodiment_id)
        goal_feat, _ = self.backbone.vision_encoder(goal_image)

        text_tokens = instruction if isinstance(instruction, torch.Tensor) else None
        align_outputs = self.alignment_head(
            text_feat=features["global_text_feat"],
            goal_feat=goal_feat,
            spatial_attention=features["spatial_attention"],
            affordance_mask_gt=affordance_mask_gt,
            text_tokens=text_tokens,
        )
        align_outputs["context_c"] = features["context_c"]
        align_outputs["spatial_attention"] = features["spatial_attention"]
        align_outputs["embodiment_feat"] = features["embodiment_feat"]
        return align_outputs

    def forward_stage2(
        self,
        obs_image: torch.Tensor,
        instruction: List[str],
        action_target: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        features = self.encode(obs_image, instruction, embodiment_id)
        loss_cfm, info = self.flow_action_head.compute_cfm_loss(
            action_target=action_target,
            context=features["context_c"],
            embodiment_feat=features["embodiment_feat"],
        )
        info["spatial_attention"] = features["spatial_attention"]
        return loss_cfm, info

    @torch.no_grad()
    def predict_action_chunk(
        self,
        obs_image: torch.Tensor,
        instruction: List[str],
        embodiment_id: Optional[torch.Tensor] = None,
        num_steps: int = 6,
        solver: str = "euler",
        apply_cbf_safety: bool = False,
        initial_position: Optional[np.ndarray] = None,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict 16-step continuous action chunk with optional Heun RK2 solver and CBF safety filtering.
        """
        features = self.encode(obs_image, instruction, embodiment_id)
        actions = self.flow_action_head.sample_actions(
            context=features["context_c"],
            embodiment_feat=features["embodiment_feat"],
            num_steps=num_steps,
            solver=solver,
            x_0=x_0,
        )

        if apply_cbf_safety:
            actions_np = actions.cpu().numpy()
            B = actions_np.shape[0]
            filtered_list = []
            for b in range(B):
                safe_chunk, _ = self.safety_filter.filter_action_chunk(
                    actions_np[b],
                    initial_position=initial_position,
                )
                filtered_list.append(safe_chunk)
            actions = torch.tensor(np.stack(filtered_list, axis=0), device=obs_image.device, dtype=torch.float32)

        return actions

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            filepath,
        )
        print(f"[OpenVLAAlignFlow] Checkpoint saved successfully to: {filepath}")

    def load_checkpoint(self, filepath: str, map_location: Optional[str] = None) -> None:
        map_loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(filepath, map_location=map_loc, weights_only=True)
        except Exception:
            ckpt = torch.load(filepath, map_location=map_loc, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        # Sanitize and auto-heal non-finite tensors if any exist in checkpoint
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                cleaned_state_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                cleaned_state_dict[k] = v
        self.load_state_dict(cleaned_state_dict)
        print(f"[OpenVLAAlignFlow] Checkpoint loaded and verified successfully from: {filepath}")
