"""
Multi-Embodiment Robot Specifications & Physical Profiles
Covers: WidowX (BridgeData v2), Google Robot (Fractal RT-1), Franka Panda (DROID)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class EmbodimentProfile:
    """Hardware and kinetic dynamics specifications for a specific robot embodiment."""
    embodiment_id: int
    name: str
    robot_type: str  # "desktop_arm", "mobile_manipulator", "industrial_arm"
    control_freq_hz: float
    degrees_of_freedom: int
    max_cartesian_velocity: float  # m/s
    max_cartesian_acceleration: float  # m/s^2
    max_angular_velocity: float  # rad/s
    jerk_safety_threshold: float  # m/s^3
    default_dataset_key: str
    description: str

    @property
    def control_dt(self) -> float:
        return 1.0 / max(1.0, float(self.control_freq_hz))

    @property
    def max_velocity(self) -> float:
        return self.max_cartesian_velocity

    @property
    def max_acceleration(self) -> float:
        return self.max_cartesian_acceleration

    @property
    def max_jerk(self) -> float:
        return self.jerk_safety_threshold


EMBODIMENT_REGISTRY: Dict[str, EmbodimentProfile] = {
    "widowx": EmbodimentProfile(
        embodiment_id=0,
        name="WidowX 250s (BridgeData v2)",
        robot_type="desktop_arm",
        control_freq_hz=10.0,
        degrees_of_freedom=7,
        max_cartesian_velocity=0.85,
        max_cartesian_acceleration=3.5,
        max_angular_velocity=2.0,
        jerk_safety_threshold=25.0,
        default_dataset_key="bridge_dataset",
        description="Low-cost desktop manipulator for tabletop pick-and-place and wiping.",
    ),
    "google_robot": EmbodimentProfile(
        embodiment_id=1,
        name="Google Robot (Fractal20220817 / RT-1)",
        robot_type="mobile_manipulator",
        control_freq_hz=10.0,
        degrees_of_freedom=7,
        max_cartesian_velocity=1.0,
        max_cartesian_acceleration=4.0,
        max_angular_velocity=2.5,
        jerk_safety_threshold=30.0,
        default_dataset_key="fractal20220817_data",
        description="Mobile wheeled base with 7-DoF arm for long-horizon drawer/countertop tasks.",
    ),
    "franka_panda": EmbodimentProfile(
        embodiment_id=2,
        name="Franka Emika Panda (DROID-100)",
        robot_type="industrial_arm",
        control_freq_hz=20.0,
        degrees_of_freedom=7,
        max_cartesian_velocity=1.2,
        max_cartesian_acceleration=5.0,
        max_angular_velocity=3.0,
        jerk_safety_threshold=20.0,
        default_dataset_key="droid_100",
        description="High-precision 7-DoF research arm for fine multi-scene contact manipulation.",
    ),
}


def get_embodiment_profile(identifier: Any) -> EmbodimentProfile:
    """Retrieve profile by embodiment name or ID."""
    if isinstance(identifier, int):
        for prof in EMBODIMENT_REGISTRY.values():
            if prof.embodiment_id == identifier:
                return prof
        return EMBODIMENT_REGISTRY["widowx"]
    
    key = str(identifier).lower()
    for name, prof in EMBODIMENT_REGISTRY.items():
        if name in key or prof.default_dataset_key in key:
            return prof
    return EMBODIMENT_REGISTRY["widowx"]
