"""
Offline Benchmark Evaluation Package for OpenVLA-AlignFlow (4-Dimensional Physical Edition)
"""
from .offline_benchmark import (
    OfflineBenchmarkEvaluator,
    PhysicalBenchmarkMetrics,
    EmbodimentDetailedScore,
    # Backward compatibility aliases
    PhysicalBenchmarkMetrics as BenchmarkMetrics,
    EmbodimentDetailedScore as EmbodimentBenchmarkScore,
)
from .metrics.geometry_metrics import (
    compute_geometry_metrics,
    compute_hausdorff_distance,
    compute_contact_offset_distance,
)
from .metrics.physics_metrics import (
    compute_physics_metrics,
    compute_resonance_energy_ratio,
    compute_contact_momentum_surge,
    compute_manipulability_index,
)
from .metrics.temporal_metrics import compute_dtw_distance, compute_kendall_tau_subgoal

__all__ = [
    "OfflineBenchmarkEvaluator",
    "PhysicalBenchmarkMetrics",
    "EmbodimentDetailedScore",
    "BenchmarkMetrics",
    "EmbodimentBenchmarkScore",
    "compute_geometry_metrics",
    "compute_hausdorff_distance",
    "compute_contact_offset_distance",
    "compute_physics_metrics",
    "compute_resonance_energy_ratio",
    "compute_contact_momentum_surge",
    "compute_manipulability_index",
    "compute_dtw_distance",
    "compute_kendall_tau_subgoal",
]
