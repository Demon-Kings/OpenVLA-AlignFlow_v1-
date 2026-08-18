"""
7-DoF EEF Action Space Canonicalization & Quantile Normalization
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple


class ActionCanonicalizer:
    """
    Standardizes robotic actions into 7-DoF EEF Delta Pose representations:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Applies empirical quantile normalization to map actions robustly to [-1.0, 1.0].
    """

    def __init__(
        self,
        q_low: float = 0.01,
        q_high: float = 0.99,
        eps: float = 1e-6,
        q_mins: Optional[np.ndarray] = None,
        q_maxs: Optional[np.ndarray] = None,
    ):
        self.q_low = q_low
        self.q_high = q_high
        self.eps = eps
        self.q_mins = np.array(q_mins, dtype=np.float32) if q_mins is not None else None
        self.q_maxs = np.array(q_maxs, dtype=np.float32) if q_maxs is not None else None

        # Default fallback standard bounds if not fitted
        if self.q_mins is None or self.q_maxs is None:
            self._set_default_bounds()

    def _set_default_bounds(self) -> None:
        """Standard default physical bounds for desktop manipulator teleoperation."""
        # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        self.q_mins = np.array([-0.05, -0.05, -0.05, -0.15, -0.15, -0.15, 0.0], dtype=np.float32)
        self.q_maxs = np.array([0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 1.0], dtype=np.float32)

    def fit(self, action_dataset: Union[np.ndarray, list]) -> Dict[str, Any]:
        """
        Fit quantile bounds on all concatenated actions in the training set.
        Args:
            action_dataset: array of shape (N, 7) or list of trajectory action arrays
        """
        if isinstance(action_dataset, list):
            all_actions = np.concatenate([np.asarray(a, dtype=np.float32) for a in action_dataset], axis=0)
        else:
            all_actions = np.asarray(action_dataset, dtype=np.float32)

        if all_actions.ndim != 2 or all_actions.shape[1] < 7:
            raise ValueError(f"Expected action shape (N, 7+), got {all_actions.shape}")

        actions_7d = all_actions[:, :7]

        # Compute empirical quantiles per dimension
        self.q_mins = np.quantile(actions_7d, self.q_low, axis=0).astype(np.float32)
        self.q_maxs = np.quantile(actions_7d, self.q_high, axis=0).astype(np.float32)

        # Handle zero-variance dimensions (e.g. static gripper)
        diff = self.q_maxs - self.q_mins
        for i in range(len(diff)):
            if diff[i] < 1e-4:
                self.q_mins[i] -= 0.1
                self.q_maxs[i] += 0.1

        return {
            "q_mins": self.q_mins.tolist(),
            "q_maxs": self.q_maxs.tolist(),
            "q_low": self.q_low,
            "q_high": self.q_high,
        }

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize 7-DoF raw action to [-1.0, 1.0].
        Formula: 2 * (a - q_min) / (q_max - q_min + eps) - 1.0
        """
        act = np.asarray(action, dtype=np.float32)
        target_shape = act.shape
        act_flat = act.reshape(-1, 7)

        denom = self.q_maxs - self.q_mins + self.eps
        normalized = 2.0 * (act_flat[:, :7] - self.q_mins) / denom - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.reshape(target_shape)

    def denormalize(self, norm_action: np.ndarray) -> np.ndarray:
        """
        Denormalize action from [-1.0, 1.0] back to physical units (meters / radians / gripper).
        Formula: (norm_a + 1.0) / 2.0 * (q_max - q_min + eps) + q_min
        """
        norm_act = np.asarray(norm_action, dtype=np.float32)
        target_shape = norm_act.shape
        norm_flat = np.clip(norm_act.reshape(-1, 7), -1.0, 1.0)

        denom = self.q_maxs - self.q_mins + self.eps
        unnorm = (norm_flat[:, :7] + 1.0) * 0.5 * denom + self.q_mins

        return unnorm.reshape(target_shape)

    def save_stats(self, filepath: str) -> None:
        """Save quantile stats to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        stats = {
            "q_mins": self.q_mins.tolist(),
            "q_maxs": self.q_maxs.tolist(),
            "q_low": self.q_low,
            "q_high": self.q_high,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

    def load_stats(self, filepath: str) -> None:
        """Load quantile stats from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.q_mins = np.array(stats["q_mins"], dtype=np.float32)
        self.q_maxs = np.array(stats["q_maxs"], dtype=np.float32)
        self.q_low = float(stats.get("q_low", 0.01))
        self.q_high = float(stats.get("q_high", 0.99))
