"""
Global Configuration Definition for OpenVLA-AlignFlow (~6GB VRAM Lightweight Edition)
"""
import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from .embodiment_configs import EMBODIMENT_REGISTRY, EmbodimentProfile, get_embodiment_profile


@dataclass
class VLAConfig:
    """
    Comprehensive Configuration for Multi-Embodiment OpenVLA-AlignFlow Pipeline:
    Lightweight ~6GB VRAM Target for RTX 4090:
      - 768-dim Vision Backbone (4 Layers, 12 Heads)
      - 768-dim Text Transformer (4 Layers, 12 Heads)
      - 512-dim Flow Matching Action Head (4 Layers)
      - Batch Size = 64 (Fast, Stable, ~5.8GB - 6.2GB VRAM Footprint)
    """
    # -------------------------------------------------------------
    # 1. Environment & Hardware (~6GB Target)
    # -------------------------------------------------------------
    device: str = "cuda"  # "cuda" (RTX 4090)
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True  # Automatic Mixed Precision (FP16)
    checkpoint_dir: str = "./checkpoints"
    pretrained_models_dir: str = "./pretrained_models"
    data_dir: str = "./data"
    output_dir: str = "./data/processed"

    # -------------------------------------------------------------
    # 2. Multi-Embodiment Configuration
    # -------------------------------------------------------------
    num_embodiments: int = 3  # 0: WidowX, 1: Google Robot, 2: Franka Panda
    embodiment_embed_dim: int = 128  # 128-dim lightweight embodiment embedding
    default_embodiment_id: int = 0

    # -------------------------------------------------------------
    # 3. Data Engineering & Canonicalization
    # -------------------------------------------------------------
    image_size: int = 224
    raw_image_size: int = 256
    action_dim: int = 7  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
    chunk_size: int = 16  # Action Chunking horizon (k=16)
    delta_t: float = 0.1  # Standard control interval (10Hz)
    
    # Kinetic Filter Thresholds
    max_velocity_threshold: float = 0.85  # m/s
    max_acceleration_threshold: float = 3.5  # m/s^2
    max_idle_ratio_threshold: float = 0.55  # 55% stationary steps
    idle_velocity_epsilon: float = 0.005  # m/s

    # Quantile Normalization Bounds
    quantile_low: float = 0.01
    quantile_high: float = 0.99

    # -------------------------------------------------------------
    # 4. Vision-Language-Action Backbone Architecture (~6GB VRAM Scale)
    # -------------------------------------------------------------
    vision_patch_size: int = 16
    vision_embed_dim: int = 768  # 768-dim (ViT-Base Standard)
    vision_num_heads: int = 12
    vision_num_layers: int = 4  # 4-Layer Fast ViT
    
    text_vocab_size: int = 10000
    text_embed_dim: int = 768  # 768-dim Text Transformer
    text_max_length: int = 32
    text_num_heads: int = 12
    text_num_layers: int = 4  # 4-Layer Text Transformer
    
    cross_attn_heads: int = 12
    projection_dim: int = 256

    # -------------------------------------------------------------
    # 5. Stage 1: Fine-Grained VL Alignment (Optimized Epochs)
    # -------------------------------------------------------------
    stage1_epochs: int = 25  # Reduced from 80 to 25 to save compute (loss plateaus at 15)
    stage1_batch_size: int = 128  # ~6GB VRAM batch size
    stage1_lr: float = 3.0e-4  # Optimized learning rate for rapid contrastive symmetry breaking
    stage1_weight_decay: float = 1e-4
    infonce_temperature: float = 0.05  # Sharpened contrastive gradient for milestone discrimination
    affordance_weight: float = 2.5  # High-priority spatial attention guidance
    affordance_temperature: float = 0.10

    # -------------------------------------------------------------
    # 6. Stage 2: Lie Group SE(3) Flow Matching (100 Epochs SOTA)
    # -------------------------------------------------------------
    flow_hidden_dim: int = 512  # 512-dim Flow Velocity Field
    flow_num_layers: int = 4  # 4-Layer Residual Flow Blocks
    time_embed_dim: int = 128
    stage2_epochs: int = 200  # 100 epochs for deep convergence (loss < 0.10)
    stage2_batch_size: int = 128  # ~6GB VRAM batch size
    stage2_lr: float = 2.5e-4  # Optimized learning rate
    stage2_weight_decay: float = 1e-5
    ode_inference_steps: int = 16  # 16-step Heun RK2 integrator for superior trajectory smoothness
    ode_eval_steps: int = 16  # 16-step evaluation

    # Lie Group SE(3) Loss Decoupling Weights
    pos_loss_weight: float = 1.0
    rot_geodesic_weight: float = 0.6  # Optimal SO(3) rotational alignment (suppresses rotation error < 20 deg)
    gripper_loss_weight: float = 0.5

    # -------------------------------------------------------------
    # 7. Stage 3: Multi-Embodiment Trajectory-DPO (Deepened: 45 Epochs)
    # -------------------------------------------------------------
    stage3_epochs: int = 100  # Deepened preference alignment
    stage3_batch_size: int = 128  # ~6GB VRAM batch size
    stage3_lr: float = 5e-5
    stage3_weight_decay: float = 1e-4
    
    # DPO Mechanism 1: KKT Dynamic Beta Dual Factor
    beta_init: float = 0.10
    beta_min: float = 0.02
    beta_max: float = 1.00  # Broader dynamic margin for preference separation
    beta_lr: float = 0.001
    beta_target_kl: float = 0.05

    # DPO Mechanism 2: Cauchy C1 Smooth BNF
    bnf_weight: float = 0.20  # Increased to strongly penalize workspace boundary violations
    bnf_margin: float = 0.15  # Widen the safety margin 
    bnf_beta: float = 10.0

    # DPO Mechanism 3: Trajectory Length Penalty & KKT Lambda Dual Factor
    length_lambda_init: float = 0.08  # Stronger idling/detour suppression (was 0.05)
    length_lambda_max: float = 0.15
    length_lambda_lr: float = 0.0005
    length_beta: float = 5.0

    # DPO Mechanism 4: SFT / BC Auxiliary Imitation Loss
    bc_aux_weight: float = 0.05

    # DPO Mechanism 5: Riemannian Geodesic & Contact-Aware Energy Damping
    riemann_weight: float = 0.005
    energy_damping_weight: float = 1.5  # EXTREME kinetic shock damping for noisy Google Robot data
    use_hutchinson_trace: bool = True  # Enable Monte Carlo Hutchinson divergence estimation
    hutchinson_weight: float = 0.01

    # -------------------------------------------------------------
    # 8. Benchmark & 7-Dimensional Physics Evaluation
    # -------------------------------------------------------------
    eval_num_samples: int = 500  # Evaluates up to 500 samples covering all 3 embodiments
    jerk_safety_line: float = 25.0  # m/s^3 real robot physical threshold
    affordance_iou_quantile: float = 0.70
    entropy_num_rollouts: int = 6
    dtw_radius: int = 10
    max_geodesic_rot_error_threshold: float = 0.08  # rad

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLAConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "VLAConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_config() -> VLAConfig:
    return VLAConfig()
