"""
Lie Group SO(3) / SE(3) Differential Geometry & Geodesic Distance Computations
Provides:
  - Euler Angles (XYZ) to SO(3) 3x3 Rotation Matrices
  - Matrix Exponential & Logarithm Mappings on Lie Group SO(3) / Lie Algebra so(3)
  - Geodesic Riemannian SLERP Interpolation: R_t = R_0 Exp(t Log(R_0^T R_1))
  - Geodesic Riemannian Distance on SO(3): d(R1, R2) = arccos((Tr(R1 R2^T) - 1) / 2)
  - Quaternion <-> Rotation Matrix <-> Euler Angles Bidirectional Conversions
  - Decoupled SE(3) Metric Tensor Computations (Translation + Rotation)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Optional


def euler_angles_to_rotation_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw in radians) to 3x3 SO(3) Rotation Matrices (XYZ convention).
    Args:
        euler_angles: (..., 3) [roll, pitch, yaw]
    Returns:
        R: (..., 3, 3) SO(3) orthogonal rotation matrices
    """
    roll = euler_angles[..., 0]
    pitch = euler_angles[..., 1]
    yaw = euler_angles[..., 2]

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # Rx * Ry * Rz (XYZ convention)
    r00 = cp * cy
    r01 = -cp * sy
    r02 = sp
    r10 = cr * sy + sr * sp * cy
    r11 = cr * cy - sr * sp * sy
    r12 = -sr * cp
    r20 = sr * sy - cr * sp * cy
    r21 = sr * cy + cr * sp * sy
    r22 = cr * cp

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)

    R = torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)
    return R


def rotation_matrix_to_euler_angles(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 SO(3) Rotation Matrices to Euler angles [roll, pitch, yaw] (XYZ convention).
    Args:
        R: (..., 3, 3)
    Returns:
        euler: (..., 3) [roll, pitch, yaw] in radians
    """
    # Clamp R[..., 0, 2] to [-1.0, 1.0] for arcsin safety
    sp = torch.clamp(R[..., 0, 2], min=-0.999999, max=0.999999)
    pitch = torch.asin(sp)
    
    # roll = atan2(-R[1, 2], R[2, 2])
    roll = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    # yaw = atan2(-R[0, 1], R[0, 0])
    yaw = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    
    return torch.stack([roll, pitch, yaw], dim=-1)


def hat_operator_so3(omega: torch.Tensor) -> torch.Tensor:
    """
    Converts 3D Lie algebra vector omega in R^3 to skew-symmetric matrix [omega]_x in so(3).
    """
    w1 = omega[..., 0]
    w2 = omega[..., 1]
    w3 = omega[..., 2]
    zeros = torch.zeros_like(w1)

    row0 = torch.stack([zeros, -w3, w2], dim=-1)
    row1 = torch.stack([w3, zeros, -w1], dim=-1)
    row2 = torch.stack([-w2, w1, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp_map(omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Rodrigues formula: Maps Lie Algebra vector omega in so(3) to SO(3) rotation matrix.
    Exp(omega) = I + (sin(theta)/theta) [omega]_x + ((1 - cos(theta))/theta^2) [omega]_x^2
    """
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)  # (..., 1)
    theta_sq = theta ** 2
    
    # Safe series expansion for small angles
    small_angle = theta < eps
    c1 = torch.where(small_angle, 1.0 - theta_sq / 6.0, torch.sin(theta) / (theta + 1e-12))
    c2 = torch.where(small_angle, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_sq + 1e-12))
    
    K = hat_operator_so3(omega)  # (..., 3, 3)
    K2 = torch.matmul(K, K)      # (..., 3, 3)
    
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(K)
    c1_mat = c1.unsqueeze(-1)
    c2_mat = c2.unsqueeze(-1)
    
    R = I + c1_mat * K + c2_mat * K2
    return R


def so3_log_map(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Logarithmic map: Maps SO(3) rotation matrix to Lie Algebra tangent vector omega in so(3) ~ R^3.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...,)
    
    # Skew symmetric part: (R - R^T) / 2
    skew = (R - R.transpose(-1, -2)) * 0.5
    w1 = skew[..., 2, 1]
    w2 = skew[..., 0, 2]
    w3 = skew[..., 1, 0]
    unscaled_omega = torch.stack([w1, w2, w3], dim=-1)
    
    sin_theta = torch.sin(theta).unsqueeze(-1)
    small_angle = (theta < eps).unsqueeze(-1)
    scale = torch.where(small_angle, 1.0 + theta.unsqueeze(-1)**2 / 6.0, theta.unsqueeze(-1) / (sin_theta + 1e-12))
    
    return unscaled_omega * scale


def so3_geodesic_interp(R0: torch.Tensor, R1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    True Riemannian Geodesic SLERP on Lie Group SO(3):
    R_t = R0 * Exp(t * Log(R0^T * R1))
    """
    R_rel = torch.matmul(R0.transpose(-1, -2), R1)
    omega_rel = so3_log_map(R_rel)  # (..., 3)
    
    if isinstance(t, (int, float)):
        t_tensor = torch.tensor(t, dtype=R0.dtype, device=R0.device)
    else:
        t_tensor = t
    
    while t_tensor.ndim < omega_rel.ndim:
        t_tensor = t_tensor.unsqueeze(-1)
        
    omega_t = t_tensor * omega_rel
    R_t_rel = so3_exp_map(omega_t)
    return torch.matmul(R0, R_t_rel)


def compute_so3_geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """
    Compute geodesic distance (minimal arc angle in radians) on the SO(3) Riemannian manifold:
    d(R1, R2) = arccos( clamp((Tr(R1 R2^T) - 1) / 2, -1 + eps, 1 - eps) )
    Args:
        R1: (..., 3, 3)
        R2: (..., 3, 3)
        eps: safety clamping margin
    Returns:
        theta: (...,) angular distance in radians in [0, pi]
    """
    # Upcast to float32 to prevent FP16 precision loss and acos derivative singularity
    R1_32 = R1.float()
    R2_32 = R2.float()

    # Relative rotation: R_rel = R1 @ R2.transpose(-1, -2)
    R_rel = torch.matmul(R1_32, R2_32.transpose(-1, -2))
    # Matrix trace: Tr(R_rel) = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta_clamped = torch.clamp(cos_theta, min=-0.999, max=0.999)
    theta = torch.acos(cos_theta_clamped)
    return theta.to(dtype=R1.dtype)


def compute_se3_action_geodesic_error(
    action_pred: torch.Tensor,
    action_gt: torch.Tensor,
    pos_weight: float = 1.0,
    rot_weight: float = 1.0,
    gripper_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute decoupled SE(3) action error:
      - Positional L2 error on R^3
      - Geodesic angular error on SO(3)
      - Gripper L1 error
    Args:
        action_pred: (B, 16, 7) or (B, 7)
        action_gt: (B, 16, 7) or (B, 7)
    Returns:
        (total_se3_error, mean_pos_error, mean_rot_geodesic_rad)
    """
    pos_pred = action_pred[..., :3]
    pos_gt = action_gt[..., :3]
    pos_error = F.mse_loss(pos_pred, pos_gt)

    rot_euler_pred = action_pred[..., 3:6]
    rot_euler_gt = action_gt[..., 3:6]

    R_pred = euler_angles_to_rotation_matrix(rot_euler_pred)
    R_gt = euler_angles_to_rotation_matrix(rot_euler_gt)

    rot_geodesic = compute_so3_geodesic_distance(R_pred, R_gt)
    mean_rot_error = rot_geodesic.mean()

    gripper_pred = action_pred[..., 6]
    gripper_gt = action_gt[..., 6]
    gripper_error = F.l1_loss(gripper_pred, gripper_gt)

    total_error = pos_weight * pos_error + rot_weight * mean_rot_error + gripper_weight * gripper_error
    return total_error, pos_error, mean_rot_error
