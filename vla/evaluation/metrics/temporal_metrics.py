"""
Temporal Causality, Dynamic Time Warping (DTW), and Causal Kendall's Tau Metrics
"""
import numpy as np
from typing import Dict, Any, List


def compute_dtw_distance(seq_pred: np.ndarray, seq_gt: np.ndarray) -> float:
    """
    Computes Dynamic Time Warping (DTW) distance between two time-series sequences.
    Args:
        seq_pred: (T1, D)
        seq_gt: (T2, D)
    Returns:
        normalized_dtw_distance: float
    """
    T1, D = seq_pred.shape
    T2, _ = seq_gt.shape

    # Pairwise Euclidean cost matrix
    cost_matrix = np.zeros((T1, T2), dtype=np.float32)
    for i in range(T1):
        for j in range(T2):
            cost_matrix[i, j] = np.linalg.norm(seq_pred[i] - seq_gt[j])

    # Accumulated cost matrix
    dtw = np.full((T1 + 1, T2 + 1), fill_value=np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    normalized_dtw = dtw[T1, T2] / (T1 + T2)
    return float(normalized_dtw)


def compute_kendall_tau_subgoal(pred_scores: np.ndarray, gt_stages: np.ndarray) -> float:
    """
    Computes Causal Kendall's Tau Rank Correlation for sub-goal progression ordering:
    tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U)) in [-1, +1]
    """
    n = len(pred_scores)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = pred_scores[i] - pred_scores[j]
            gt_diff = gt_stages[i] - gt_stages[j]
            prod = pred_diff * gt_diff
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0
    tau = (concordant - discordant) / total_pairs
    return float(np.clip(tau, -1.0, 1.0))
