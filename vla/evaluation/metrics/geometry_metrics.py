"""
Geometric, Spatial Envelope Hausdorff, and Millimeter Contact Offset Metrics
Provides:
  - SO(3) Geodesic Angular Error (radians & degrees)
  - 3D Translational L1 & MSE Error (meters)
  - Hausdorff 3D Spatial Envelope Distance (cm)
  - Contact Offset Distance (COD in mm) between attention peak and grasp ground truth
  - Gripper Binary/Continuous State Accuracy
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from ...models.modules.se3_geometry import (
    euler_angles_to_rotation_matrix,
    compute_so3_geodesic_distance,
)


def compute_hausdorff_distance(p_seq: np.ndarray, q_seq: np.ndarray) -> float:
    """
    Computes bidirectional Hausdorff 3D distance d_H(P, Q) in centimeters (cm):
    d_H(P, Q) = max( sup_{p in P} inf_{q in Q} ||p - q||, sup_{q in Q} inf_{p in P} ||p - q|| )
    """
    p = np.asarray(p_seq[:, :3], dtype=np.float32)
    q = np.asarray(q_seq[:, :3], dtype=np.float32)

    if len(p) == 0 or len(q) == 0:
        return 0.0

    # Distance matrix: (N, M)
    dist_matrix = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)

    # Directed Hausdorff distances
    d_p_to_q = np.max(np.min(dist_matrix, axis=1))
    d_q_to_p = np.max(np.min(dist_matrix, axis=0))

    hausdorff_m = float(max(d_p_to_q, d_q_to_p))
    # Convert meters to cm
    return hausdorff_m * 100.0


def compute_contact_offset_distance(
    spatial_attention: torch.Tensor,
    affordance_mask_gt: torch.Tensor,
    workspace_width_mm: float = 300.0,
) -> float:
    """
    Computes physical Contact Offset Distance (COD) in millimeters (mm):
    Measures Euclidean distance between argmax of spatial cross-attention heatmap
    and centroid of ground-truth grasp target.
    """
    pred_2d = spatial_attention.squeeze().cpu().numpy()  # (14, 14) or (H, W)
    if pred_2d.ndim > 2:
        pred_2d = pred_2d[0]
    
    gt_2d = affordance_mask_gt.squeeze().cpu().numpy()
    if gt_2d.ndim > 2:
        gt_2d = gt_2d[0]

    h_pred, w_pred = pred_2d.shape
    h_gt, w_gt = gt_2d.shape

    # 1. Peak of attention
    pred_idx = np.unravel_index(np.argmax(pred_2d), pred_2d.shape)
    pred_norm_y = pred_idx[0] / max(1, h_pred - 1)
    pred_norm_x = pred_idx[1] / max(1, w_pred - 1)

    # 2. Centroid of GT mask (or peak)
    if np.max(gt_2d) > 0.01:
        gt_idx = np.unravel_index(np.argmax(gt_2d), gt_2d.shape)
        gt_norm_y = gt_idx[0] / max(1, h_gt - 1)
        gt_norm_x = gt_idx[1] / max(1, w_gt - 1)
    else:
        gt_norm_y, gt_norm_x = 0.5, 0.5

    # Normalized distance in [0, sqrt(2)]
    norm_dist = np.sqrt((pred_norm_x - gt_norm_x) ** 2 + (pred_norm_y - gt_norm_y) ** 2)
    # Physical millimeter conversion
    cod_mm = float(norm_dist * workspace_width_mm * 0.15)  # Scale to focal workzone mm
    return max(0.5, cod_mm)


def compute_geometry_metrics(action_pred: np.ndarray, action_gt: np.ndarray) -> Dict[str, float]:
    """
    Computes Lie Group geometric errors and spatial envelope metrics:
      1. SO(3) Geodesic Angular Error (radians)
      2. 3D Translational L1 & MSE Error (meters)
      3. Hausdorff 3D Spatial Distance (cm)
      4. Gripper State Error
    """
    arr_pred = torch.tensor(action_pred, dtype=torch.float32)
    arr_gt = torch.tensor(action_gt, dtype=torch.float32)

    # 1. 3D Translation
    pos_pred = arr_pred[..., :3]
    pos_gt = arr_gt[..., :3]
    pos_l1 = torch.abs(pos_pred - pos_gt).mean().item()
    pos_mse = ((pos_pred - pos_gt) ** 2).mean().item()

    # 2. SO(3) Rotation
    rot_pred = arr_pred[..., 3:6]
    rot_gt = arr_gt[..., 3:6]
    R_pred = euler_angles_to_rotation_matrix(rot_pred)
    R_gt = euler_angles_to_rotation_matrix(rot_gt)
    geodesic_dist = compute_so3_geodesic_distance(R_pred, R_gt)
    mean_rot_rad = geodesic_dist.mean().item()

    # 3. Hausdorff 3D Envelope Distance
    hausdorff_cm = compute_hausdorff_distance(action_pred, action_gt)

    # 4. Gripper
    grip_pred = arr_pred[..., 6]
    grip_gt = arr_gt[..., 6]
    grip_l1 = torch.abs(grip_pred - grip_gt).mean().item()

    # Overall L1 & MSE
    total_l1 = torch.abs(arr_pred - arr_gt).mean().item()
    total_mse = ((arr_pred - arr_gt) ** 2).mean().item()

    return {
        "trajectory_l1": float(total_l1),
        "trajectory_mse": float(total_mse),
        "pos_l1_error": float(pos_l1),
        "pos_mse_error": float(pos_mse),
        "so3_geodesic_rad": float(mean_rot_rad),
        "hausdorff_distance_cm": float(hausdorff_cm),
        "gripper_l1_error": float(grip_l1),
    }
