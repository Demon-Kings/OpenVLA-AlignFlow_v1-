"""
Kinetic Jitter & Dynamics Anomaly Detection Filter for Embodied Teleoperation Trajectories
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional


@dataclass
class KineticMetrics:
    """Calculated kinetic dynamics metrics for a single trajectory."""
    max_velocity: float
    mean_velocity: float
    max_acceleration: float
    mean_acceleration: float
    idle_ratio: float
    is_expert: bool
    rejection_reason: Optional[str] = None


class KineticJitterFilter:
    """
    Kinetic Jitter Filter:
    Inspects 1st order velocity, 2nd order acceleration, and idle step ratio of robot trajectories.
    Separates raw teleoperated records into:
      1. Gold Expert Trajectories (High-fidelity, smooth, goal-directed)
      2. Noisy / Rejected Trajectories (Saved as negative samples for Trajectory-DPO)
    """

    def __init__(
        self,
        max_velocity_threshold: float = 0.85,
        max_acceleration_threshold: float = 3.5,
        max_idle_ratio_threshold: float = 0.55,
        idle_velocity_epsilon: float = 0.005,
        delta_t: float = 0.1,
    ):
        self.v_max_thresh = max_velocity_threshold
        self.a_max_thresh = max_acceleration_threshold
        self.idle_thresh = max_idle_ratio_threshold
        self.idle_eps = idle_velocity_epsilon
        self.dt = delta_t

    def compute_kinetics(self, positions_or_actions: np.ndarray) -> KineticMetrics:
        """
        Compute kinetic metrics from 3D positions or 7D actions.
        Args:
            positions_or_actions: (L, 3) or (L, 7) array. If (L, 7), first 3 dims are [dx, dy, dz].
        Returns:
            KineticMetrics object
        """
        arr = np.asarray(positions_or_actions, dtype=np.float32)
        if arr.ndim != 2 or len(arr) < 3:
            return KineticMetrics(
                max_velocity=0.0,
                mean_velocity=0.0,
                max_acceleration=0.0,
                mean_acceleration=0.0,
                idle_ratio=1.0,
                is_expert=False,
                rejection_reason="Trajectory too short or invalid shape",
            )

        # Extract 3D translational component
        if arr.shape[1] >= 3:
            xyz = arr[:, :3]
        else:
            raise ValueError(f"Expected at least 3 dimensions for positions/actions, got {arr.shape[1]}")

        # Compute velocity magnitudes: ||p_{t+1} - p_t|| / dt
        diff_pos = np.diff(xyz, axis=0)  # (L-1, 3)
        velocities = np.linalg.norm(diff_pos, axis=1) / self.dt  # (L-1,)

        # Compute acceleration magnitudes: ||v_{t+1} - v_t|| / dt
        diff_vel = np.diff(diff_pos / self.dt, axis=0)  # (L-2, 3)
        accelerations = np.linalg.norm(diff_vel, axis=1) / self.dt  # (L-2,)

        max_v = float(np.max(velocities)) if len(velocities) > 0 else 0.0
        mean_v = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
        max_a = float(np.max(accelerations)) if len(accelerations) > 0 else 0.0
        mean_a = float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0

        # Calculate idle ratio
        idle_count = np.sum(velocities < self.idle_eps)
        idle_ratio = float(idle_count / max(len(velocities), 1))

        # Multi-stage Kinetic Decision Logic
        is_expert = True
        rejection_reason = None

        if max_v > self.v_max_thresh:
            is_expert = False
            rejection_reason = f"Max velocity ({max_v:.3f} m/s) exceeded threshold ({self.v_max_thresh} m/s)"
        elif max_a > self.a_max_thresh:
            is_expert = False
            rejection_reason = f"Max acceleration ({max_a:.3f} m/s^2) exceeded threshold ({self.a_max_thresh} m/s^2)"
        elif idle_ratio > self.idle_thresh:
            is_expert = False
            rejection_reason = f"Idle ratio ({idle_ratio:.1%}) exceeded threshold ({self.idle_thresh:.1%})"

        return KineticMetrics(
            max_velocity=max_v,
            mean_velocity=mean_v,
            max_acceleration=max_a,
            mean_acceleration=mean_a,
            idle_ratio=idle_ratio,
            is_expert=is_expert,
            rejection_reason=rejection_reason,
        )

    def filter_dataset(
        self, trajectories: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter a list of raw episode trajectory dictionaries.
        Args:
            trajectories: list of dicts with key 'actions' or 'positions'
        Returns:
            (expert_trajectories, noisy_rejected_trajectories, filter_stats)
        """
        expert_trajs = []
        noisy_trajs = []
        stats = {
            "total_input": len(trajectories),
            "expert_count": 0,
            "rejected_count": 0,
            "reject_by_velocity": 0,
            "reject_by_acceleration": 0,
            "reject_by_idle": 0,
            "expert_ratio": 0.0,
        }

        for traj in trajectories:
            data = traj.get("actions", traj.get("positions"))
            if data is None:
                continue
            metrics = self.compute_kinetics(data)
            traj_copy = dict(traj)
            traj_copy["kinetic_metrics"] = metrics

            if metrics.is_expert:
                traj_copy["is_expert"] = True
                traj_copy["return"] = 1.0
                expert_trajs.append(traj_copy)
                stats["expert_count"] += 1
            else:
                traj_copy["is_expert"] = False
                traj_copy["return"] = 0.15
                noisy_trajs.append(traj_copy)
                stats["rejected_count"] += 1

                reason = metrics.rejection_reason or ""
                if "velocity" in reason:
                    stats["reject_by_velocity"] += 1
                elif "acceleration" in reason:
                    stats["reject_by_acceleration"] += 1
                elif "Idle" in reason or "idle" in reason:
                    stats["reject_by_idle"] += 1

        if stats["total_input"] > 0:
            stats["expert_ratio"] = stats["expert_count"] / stats["total_input"]

        return expert_trajs, noisy_trajs, stats
