"""
Core Mathematical & Physical Submodules Package
"""
from .se3_geometry import (
    euler_angles_to_rotation_matrix,
    compute_so3_geodesic_distance,
    compute_se3_action_geodesic_error,
)
from .embodiment_encoder import EmbodimentEmbedding, FiLMLayer
from .safety_cbf import KinematicCBFSafetyFilter

__all__ = [
    "euler_angles_to_rotation_matrix",
    "compute_so3_geodesic_distance",
    "compute_se3_action_geodesic_error",
    "EmbodimentEmbedding",
    "FiLMLayer",
    "KinematicCBFSafetyFilter",
]
