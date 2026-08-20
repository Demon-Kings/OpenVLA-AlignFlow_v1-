<div align="center">

# 🤖 OpenVLA-AlignFlow

**Continuous Flow Matching & Trajectory-DPO for Multi-Embodiment Robots**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-1x_RTX_4090_(24GB)-76B900.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#) | [中文技术手册](./OpenVLA_AlignFlow_全架构全流程终极技术手册_20260818_101152.md)

</div>

**OpenVLA-AlignFlow** is an end-to-end, multi-embodiment Vision-Language-Action (VLA) framework designed to bring foundation models to physical robots. By replacing traditional Diffusion processes with **Optimal Transport Conditional Flow Matching (OT-CFM)** and introducing **Trajectory-DPO** for physical alignment, it generates extremely smooth, safe, and resonance-free actions. 

Crucially, the entire training, auto-tuning, and evaluation pipeline is heavily engineered to run natively on a **single consumer-grade RTX 4090 (24GB VRAM)**.

---

## ✨ Key Features

*   ⚡ **Optimal Transport Flow Matching (OT-CFM):** Generates actions along straight-line vector fields. Requires fewer ODE solver steps (16 steps) and completely eliminates the high-frequency jitter (Jerk) associated with DDPM/DDIM.
*   🎯 **Physical Trajectory-DPO:** Reinforcement Learning from Human Feedback (RLHF) adapted for physics. Uses a Contact-Aware Energy Damping penalty to suppress 12~25Hz resonance (RER) and ensure soft landings.
*   🛡️ **Zero-Latency CBF Safety Filter:** A deterministic, closed-form Control Barrier Function (CBF) tensor broadcast solver that intercepts out-of-bounds workspace violations in **12.4 microseconds**.
*   🌐 **Multi-Embodiment Support:** Natively supports Google Robot (RT-1), WidowX (BridgeData v2), and Franka Panda (DROID) out of the box.
*   💻 **Consumer GPU Optimized:** Zero-copy PCIe bypass and heavily optimized 768-dim ViT/Text backbones keep peak VRAM under 24GB.

## 🚀 The Three-Stage Training Architecture

1.  **Stage 1: Vision-Language Alignment** (Contrastive InfoNCE + Affordance KL). Breaks multimodal symmetry and establishes spatial heatmaps.
2.  **Stage 2: SE(3) Flow Matching** (OT-CFM). Trains the base continuous action distribution vector fields.
3.  **Stage 3: Trajectory-DPO** (RLHF). Fine-tunes the flow head using physical preference pairs to guarantee smooth, safe, and highly reliable grasping dynamics.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/OpenVLA-AlignFlow.git
cd OpenVLA-AlignFlow

# Create a conda environment
conda create -n vla_alignflow python=3.10 -y
conda activate vla_alignflow

# Install dependencies (requires PyTorch with CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🗄️ Dataset Preparation (`dddd.py`)

We provide a **Production-Grade Batch Downloader** (`dddd.py`) to easily fetch massive open-source robotic datasets (BridgeData v2, RT-1, DROID) from Google Cloud Storage.

```bash
# Download all 3 core datasets concurrently (Default: 10 shards per dataset)
python dddd.py --dataset all --max_shards 10 --workers 8

# Output directories created automatically:
# - ./bridge_dataset/
# - ./fractal20220817_data/
# - ./droid_100/
```

---

## 🏃 Quick Start: Training & Evaluation

The entire pipeline is wrapped in a single entry point: `run_pipeline.py`.

```bash
# 1. Run the full End-to-End Pipeline (Stage 1 -> Stage 2 -> Stage 3 -> Eval)
python run_pipeline.py --mode full

# 2. Run a specific training stage (e.g., Stage 3 Trajectory-DPO only)
python run_pipeline.py --start_stage 3

# 3. Run the 4-Dimensional Physical Offline Benchmark
python run_pipeline.py --start_stage eval --eval_num_samples 500
```

---

## 🧬 Advanced: Auto-Tuning PID Engine

Tuning RL/DPO parameters by hand is tedious. OpenVLA-AlignFlow includes a standalone **Dynamic PID Scheduling Engine** (`dynamic_pid_engine.py`) designed specifically for a single RTX 4090.

It chunks the training process, evaluates the physical metrics (Jerk, RER, Recall, Workspace Violation, COD), calculates a **Comprehensive Fitness Score**, and uses a PID controller to dynamically update `energy_damping_weight` in real-time.

```bash
# Start the auto-tuning evolutionary engine
python dynamic_pid_engine.py
```
*   **Traceability**: Fully logs all 20 benchmark metrics to `checkpoints/pid_tuning_history_full.csv`.
*   **Champion Checkpoint**: Automatically locks and saves the absolute best configuration to `checkpoints/best_pid_parameters.json`.

---

## 📊 Benchmark Results

Evaluated on 500 zero-shot, multi-embodiment rollouts on the latest optimized parameters:

| Metric Category | Metric | Result | Target / Limit |
| :--- | :--- | :--- | :--- |
| **Cognitive Task** | Sub-Goal Recall R@1 | **55.00%** | Higher is better |
| **Spatial Precision**| Contact Offset Distance (COD)| **1.63 mm** | < 2.0 mm (SOTA) |
| **Safety** | CBF Safety Barrier Margin | **+0.0024 m** | > 0.0 m |
| **Physical Health** | Physical Mean Jerk | **9.49 m/s³** | < 25.0 m/s³ |

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
