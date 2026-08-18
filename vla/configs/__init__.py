"""
Configuration Package for OpenVLA-AlignFlow (Multi-Embodiment)
"""
from .config import VLAConfig, get_default_config
from .embodiment_configs import EMBODIMENT_REGISTRY, EmbodimentProfile, get_embodiment_profile

__all__ = [
    "VLAConfig",
    "get_default_config",
    "EMBODIMENT_REGISTRY",
    "EmbodimentProfile",
    "get_embodiment_profile",
]
