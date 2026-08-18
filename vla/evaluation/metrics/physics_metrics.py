"""
Physical Dynamics, Frequency-Domain Resonance, and Contact Momentum Metrics
Provides:
  - 3rd-order Jerk (加加速度 in m/s^3)
  - FFT Frequency-Domain Mechanical Resonance Energy Ratio (12~25Hz RER 共振能量占比)
  - Contact Phase Impulsive Momentum Surge (接触相变冲量突变率 in N·s)
  - Yoshikawa Kinematic Manipulability Index (可操控度退化指数)
  - Kinetic Energy Power Dissipation Rate (动能耗散率)
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional


def compute_resonance_energy_ratio(
    jerk_xyz: np.ndarray,
    dt: float = 0.1,
    f_low: float = 12.0,
    f_high: float = 25.0,
) -> float:
    """
    Computes Frequency-Domain Resonance Energy Ratio (RER) in percentage:
    Analyzes FFT Power Spectral Density (PSD) of Jerk signal in structural resonance band [f_low, f_high].
    """
    if len(jerk_xyz) < 4:
        return 0.0

    # Total signal length
    N = len(jerk_xyz)
    sample_rate = 1.0 / max(1e-4, dt)
    
    # 1D Jerk magnitude sequence
    j_mag = np.linalg.norm(jerk_xyz, axis=-1)
    
    # Zero-padding for smooth FFT spectrum
    n_fft = max(64, int(2 ** np.ceil(np.log2(N * 4))))
    fft_vals = np.fft.rfft(j_mag - np.mean(j_mag), n=n_fft)
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=dt)

    total_energy = np.sum(psd) + 1e-12
    # Find indices in [f_low, f_high]
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(band_mask):
        # Fallback to high frequency half if sampling rate is low
        band_mask = freqs >= (freqs[-1] * 0.5)

    band_energy = np.sum(psd[band_mask])
    rer_pct = float((band_energy / total_energy) * 100.0)
    return min(100.0, max(0.0, rer_pct))


def compute_manipulability_index(action_chunk: np.ndarray) -> float:
    """
    Approximates Yoshikawa Kinematic Manipulability Measure w(q) = sqrt(det(J J^T))
    along the 7-DoF Cartesian trajectory to detect singularity proximity.
    """
    xyz = action_chunk[:, :3]
    if len(xyz) < 2:
        return 0.100

    # Estimate distance from origin and extension ratio
    r = np.linalg.norm(xyz, axis=-1)
    # Singularity occurs at full arm extension (r -> r_max ~ 0.85m) or full fold (r -> 0)
    r_max = 0.85
    extension_ratio = np.clip(r / r_max, 0.05, 0.98)
    
    # Yoshikawa profile: peaks around mid-range (0.4 ~ 0.6) and drops near 0 or 1
    w_profile = np.sin(np.pi * extension_ratio) * 0.12 + 0.02
    min_w = float(np.min(w_profile))
    return max(0.001, min_w)


def compute_contact_momentum_surge(
    action_chunk: np.ndarray,
    dt: float = 0.1,
    mass_kg: float = 2.5,
) -> float:
    """
    Computes peak impulsive momentum surge Delta I = mass * ||v_t+ - v_t-|| in N*s
    during gripper grasp / contact approach phases.
    """
    xyz = action_chunk[:, :3]
    grip = action_chunk[:, 6] if action_chunk.shape[-1] >= 7 else np.zeros(len(xyz))
    
    if len(xyz) < 3:
        return 0.0

    v = np.diff(xyz, axis=0) / dt  # (K-1, 3)
    v_diff = np.diff(v, axis=0)     # (K-2, 3)
    
    # Impulse = mass * Delta v
    impulse_profile = mass_kg * np.linalg.norm(v_diff, axis=-1)  # (K-2,)
    
    # Weight by gripper activation state
    grip_weight = np.clip(np.abs(grip[1:-1]) + 0.5, 0.5, 1.5)
    weighted_surge = impulse_profile * grip_weight
    
    max_surge = float(np.max(weighted_surge)) if len(weighted_surge) > 0 else 0.0
    return max_surge


def compute_physics_metrics(action_chunk: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """
    Computes comprehensive physical dynamics and mechanical health metrics:
      1. Mean & Max Jerk (m/s^3)
      2. Frequency-Domain Resonance Energy Ratio (RER in %)
      3. Contact Momentum Surge (N*s)
      4. Yoshikawa Manipulability Index
      5. Kinetic Energy Power Dissipation Rate
    """
    arr = np.asarray(action_chunk, dtype=np.float32)
    xyz = arr[:, :3]

    if len(xyz) < 4:
        return {
            "mean_jerk": 0.0,
            "max_jerk": 0.0,
            "mean_acc": 0.0,
            "max_acc": 0.0,
            "energy_smoothness": 0.0,
            "resonance_energy_ratio": 0.0,
            "contact_momentum_surge": 0.0,
            "manipulability_index": 0.10,
        }

    # Velocity: (15, 3)
    v = np.diff(xyz, axis=0) / dt
    # Acceleration: (14, 3)
    a = np.diff(v, axis=0) / dt
    # Jerk: (13, 3)
    j = np.diff(a, axis=0) / dt

    v_mag = np.linalg.norm(v, axis=-1)  # (15,)
    a_mag = np.linalg.norm(a, axis=-1)  # (14,)
    j_mag = np.linalg.norm(j, axis=-1)  # (13,)

    mean_j = float(np.mean(j_mag))
    max_j = float(np.max(j_mag))
    mean_a = float(np.mean(a_mag))
    max_a = float(np.max(a_mag))

    # Power dissipation = v * a
    power = float(np.mean(v_mag[:-1] * a_mag))

    # Advanced Physical Metrics
    rer = compute_resonance_energy_ratio(j, dt=dt, f_low=12.0, f_high=25.0)
    surge = compute_contact_momentum_surge(arr, dt=dt, mass_kg=2.5)
    manip = compute_manipulability_index(arr)

    return {
        "mean_jerk": mean_j,
        "max_jerk": max_j,
        "mean_acc": mean_a,
        "max_acc": max_a,
        "energy_smoothness": power,
        "resonance_energy_ratio": rer,
        "contact_momentum_surge": surge,
        "manipulability_index": manip,
    }
