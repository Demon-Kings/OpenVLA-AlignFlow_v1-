"""
Stage 1: Fine-Grained Multimodal Vision-Language Alignment (Sub-goal InfoNCE + Affordance Mask KL)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union


class VLAlignmentModule(nn.Module):
    """
    Stage 1 Embodied Multimodal Alignment Module:
    1. Sub-goal InfoNCE Loss: Aligns language instructions with future goal visual milestones on the hypersphere.
    2. Affordance Mask Loss: Enforces spatial Cross-Attention to focus on manipulable object parts via KL divergence.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        projection_dim: int = 256,
        infonce_temperature: float = 0.05,
        affordance_weight: float = 1.2,
        affordance_temperature: float = 0.10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.tau = infonce_temperature
        self.aff_weight = affordance_weight
        self.aff_tau = affordance_temperature

        # Text and Goal Visual Projection Heads
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )

    def compute_infonce_loss(
        self,
        text_feat: torch.Tensor,
        goal_feat: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric Bidirectional Supervised Multi-Positive InfoNCE Contrastive Loss (SupCon).
        Computed in float32 to prevent FP16 underflow/overflow with small temperature.
        """
        # Force float32 computation for numerical precision on hypersphere
        orig_dtype = text_feat.dtype
        t_feat_f32 = text_feat.float()
        g_feat_f32 = goal_feat.float()

        z_t = F.normalize(self.text_proj(t_feat_f32), p=2, dim=-1, eps=1e-5)  # (B, proj_dim)
        z_v = F.normalize(self.goal_proj(g_feat_f32), p=2, dim=-1, eps=1e-5)  # (B, proj_dim)

        B = z_t.shape[0]
        # Logits in float32: (B, B) / tau
        sim_matrix = torch.matmul(z_t, z_v.T)  # (B, B)
        sim_matrix = torch.clamp(sim_matrix, min=-0.999, max=0.999)
        logits_tv = sim_matrix / max(float(self.tau), 1e-4)
        logits_vt = logits_tv.T

        # Construct positive instance mask across batch
        if text_tokens is not None and text_tokens.ndim == 2:
            token_prefix = text_tokens[:, :min(8, text_tokens.shape[1])]
            pos_mask = (token_prefix.unsqueeze(1) == token_prefix.unsqueeze(0)).all(dim=-1).float()
        else:
            pos_mask = torch.eye(B, device=z_t.device, dtype=torch.float32)

        # Multi-Positive Supervised Contrastive Loss (SupCon) in FP32
        log_prob_tv = F.log_softmax(logits_tv, dim=-1)
        log_prob_vt = F.log_softmax(logits_vt, dim=-1)

        pos_count = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        loss_tv = -(pos_mask * log_prob_tv).sum(dim=-1) / pos_count.squeeze(-1)
        loss_vt = -(pos_mask.T * log_prob_vt).sum(dim=-1) / pos_count.squeeze(-1)

        infonce_loss = 0.5 * (loss_tv.mean() + loss_vt.mean())
        infonce_loss = torch.nan_to_num(infonce_loss, nan=0.0, posinf=10.0, neginf=0.0).to(dtype=orig_dtype)
        return infonce_loss, sim_matrix.to(dtype=orig_dtype)

    def compute_affordance_loss(
        self,
        spatial_attention: torch.Tensor,
        affordance_mask_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Spatial Affordance Attention Mask Loss via KL Divergence and Dense Heatmap Guidance in FP32.
        """
        orig_dtype = spatial_attention.dtype
        # Ensure 4D (B, 1, H, W) and float32
        if affordance_mask_gt.dtype == torch.uint8:
            gt_f32 = affordance_mask_gt.float() / 255.0
        else:
            gt_f32 = affordance_mask_gt.float()

        if gt_f32.ndim == 2:
            gt_f32 = gt_f32.unsqueeze(0).unsqueeze(0)
        elif gt_f32.ndim == 3:
            gt_f32 = gt_f32.unsqueeze(1)

        B = spatial_attention.shape[0]
        grid_h, grid_w = spatial_attention.shape[2], spatial_attention.shape[3]
        gt_downsampled = F.interpolate(
            gt_f32,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, 14, 14)

        # 1. Float32 Cross-Attention Log-Probability KL Divergence
        p_pred = spatial_attention.float().view(B, -1)  # (B, 196)
        p_pred = p_pred / (p_pred.sum(dim=-1, keepdim=True) + 1e-8)
        log_p_pred = torch.log(torch.clamp(p_pred, min=1e-8, max=1.0))

        gt_flat = gt_downsampled.view(B, -1)
        p_gt = F.softmax(gt_flat / max(float(self.aff_tau), 1e-4), dim=-1)  # (B, 196)
        kl_loss = F.kl_div(log_p_pred, p_gt, reduction="batchmean")

        # 2. Dense Spatial Heatmap Direct Guidance
        mse_loss = F.mse_loss(p_pred * 196.0, gt_flat)
        affordance_loss = (kl_loss + 0.5 * mse_loss).to(dtype=orig_dtype)
        affordance_loss = torch.nan_to_num(affordance_loss, nan=0.0, posinf=10.0, neginf=0.0)

        return affordance_loss

    def forward(
        self,
        text_feat: torch.Tensor,
        goal_feat: torch.Tensor,
        spatial_attention: torch.Tensor,
        affordance_mask_gt: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total Stage 1 loss and sub-losses.
        """
        infonce_loss, sim_matrix = self.compute_infonce_loss(text_feat, goal_feat, text_tokens=text_tokens)
        affordance_loss = self.compute_affordance_loss(spatial_attention, affordance_mask_gt)

        total_stage1_loss = infonce_loss + self.aff_weight * affordance_loss
        total_stage1_loss = torch.nan_to_num(total_stage1_loss, nan=0.0, posinf=20.0, neginf=0.0)

        return {
            "stage1_loss": total_stage1_loss,
            "infonce_loss": infonce_loss,
            "affordance_loss": affordance_loss,
            "similarity_matrix": sim_matrix,
        }
