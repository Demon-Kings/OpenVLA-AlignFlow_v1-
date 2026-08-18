"""
Master Pipeline Orchestrator for OpenVLA-AlignFlow (RTX 4090 High-Performance Edition)
Supports:
  1. Breakpoint Resume & Fault-Tolerant Training (--resume, --resume_path, --start_stage)
  2. Isolated Timestamped Experiment Checkpoint Directories (e.g. ./checkpoints/exp_YYYYMMDD_HHMMSS/)
  3. Per-Stage Immediate Checkpoint Checkpointing (Stage 1, 2, 3 intermediate saves)
  4. Direct Benchmark Evaluation Mode (--start_stage eval)
"""
import os
import sys
import glob
import json
import argparse
import datetime
from typing import Optional, Dict, Any, List, Tuple
import torch
import numpy as np

# Ensure package root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.configs.config import VLAConfig, get_default_config
from vla.data.embodied_dataset import EmbodiedVLADataset, create_synthetic_embodied_dataset
from vla.data.canonicalize import ActionCanonicalizer
from vla.data.kinetic_filter import KineticJitterFilter
from vla.data.process_multi_openx import MultiOpenXDataProcessor
from vla.models.openvla_alignflow import OpenVLAAlignFlow
from vla.training.train_vl_align import run_stage1_vl_alignment
from vla.training.train_flow_vla import run_stage2_flow_pretraining
from vla.training.train_offline_rl_dpo import run_stage3_offline_rl_dpo
from vla.evaluation.offline_benchmark import OfflineBenchmarkEvaluator


def find_latest_checkpoint(checkpoint_root: str, target_stage: Optional[int] = None) -> Optional[str]:
    """
    Auto-detects the most recent checkpoint file across experiment runs.
    """
    if not os.path.exists(checkpoint_root):
        return None

    # Search for stage-specific checkpoints or final models
    patterns = []
    if target_stage == 2:
        patterns = ["**/stage1_checkpoint.pt", "**/stage1*.pt", "**/openvla_alignflow_final.pt"]
    elif target_stage == 3:
        patterns = ["**/stage2_checkpoint.pt", "**/stage2*.pt", "**/openvla_alignflow_final.pt"]
    elif target_stage == 4 or target_stage == "eval":
        patterns = ["**/openvla_alignflow_final.pt", "**/stage3_checkpoint.pt", "**/*.pt"]
    else:
        patterns = ["**/openvla_alignflow_final.pt", "**/stage3_checkpoint.pt", "**/stage2_checkpoint.pt", "**/stage1_checkpoint.pt", "**/*.pt"]

    candidates = []
    for pat in patterns:
        matched = glob.glob(os.path.join(checkpoint_root, pat), recursive=True)
        if matched:
            candidates.extend(matched)

    if not candidates:
        return None

    # Sort by file modification time (newest first)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def run_pipeline(
    mode: str = "full",
    device_name: str = "cuda",
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
    stage3_epochs: Optional[int] = None,
    base_dir: str = ".",
    output_dir: str = "./data/processed",
    checkpoint_dir: str = "./checkpoints",
    exp_name: Optional[str] = None,
    start_stage: str = "1",
    resume_path: Optional[str] = None,
):
    """
    Executes the entire end-to-end OpenVLA-AlignFlow pipeline with stage breakpoint resuming
    and timestamped isolated checkpoint folders.
    """
    # 1. Initialize Configuration
    cfg = get_default_config()
    cfg.device = "cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu"
    if stage1_epochs is not None:
        cfg.stage1_epochs = stage1_epochs
    if stage2_epochs is not None:
        cfg.stage2_epochs = stage2_epochs
    if stage3_epochs is not None:
        cfg.stage3_epochs = stage3_epochs
    cfg.output_dir = output_dir

    # 2. Setup Dedicated Experiment Subdirectory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = exp_name if exp_name else f"exp_{timestamp}"
    run_ckpt_dir = os.path.join(checkpoint_dir, run_folder_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    cfg.checkpoint_dir = run_ckpt_dir

    # Save Config Snapshot
    cfg.save(os.path.join(run_ckpt_dir, "vla_config.json"))

    gpu_info = "CPU Mode"
    if cfg.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"🚀 GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)"
        torch.backends.cudnn.benchmark = True

    stage_str = str(start_stage).lower()
    print("\n" + "=" * 80)
    print("🤖 OpenVLA-AlignFlow: Multi-Embodiment Embodied Decision System")
    print(f"   Hardware: {gpu_info}")
    print("   Datasets: BridgeData v2 (WidowX) + Fractal (RT-1) + DROID-100 (Franka)")
    print(f"   Mode: {mode.upper()} | Device: {cfg.device} | Horizon: k={cfg.chunk_size}")
    print(f"   Start Stage: {stage_str.upper()} | Run Directory: {run_ckpt_dir}")
    print(f"   Epochs: Stage1={cfg.stage1_epochs}, Stage2={cfg.stage2_epochs}, Stage3={cfg.stage3_epochs}")
    print("=" * 80)

    # 3. Data Preparation & Canonicalization
    if mode == "demo":
        print("\n[Step 1/5] 🛠️ Generating Multi-Embodiment Synthetic Trajectory Dataset...")
        raw_experts, raw_rejected = create_synthetic_embodied_dataset(
            num_expert_trajs=60,
            num_rejected_trajs=40,
            traj_len=40,
            image_size=cfg.image_size,
            seed=cfg.seed,
        )

        kfilter = KineticJitterFilter(
            max_velocity_threshold=cfg.max_velocity_threshold,
            max_acceleration_threshold=cfg.max_acceleration_threshold,
            max_idle_ratio_threshold=cfg.max_idle_ratio_threshold,
            delta_t=cfg.delta_t,
        )
        expert_trajs, noisy_trajs, filter_stats = kfilter.filter_dataset(raw_experts + raw_rejected)
        print(f"   Kinetic Filter: {filter_stats['expert_count']} Experts, {filter_stats['rejected_count']} Rejected.")

        canonicalizer = ActionCanonicalizer()
        canonicalizer.fit([t["actions"] for t in expert_trajs])
        for t in expert_trajs:
            t["actions"] = canonicalizer.normalize(t["actions"])
        for t in noisy_trajs:
            t["actions"] = canonicalizer.normalize(t["actions"])

        n_train = int(len(expert_trajs) * 0.80)
        train_trajs = expert_trajs[:n_train]
        test_trajs = expert_trajs[n_train:]

        train_dataset = EmbodiedVLADataset(
            trajectories=train_trajs,
            chunk_size=cfg.chunk_size,
            is_train=True,
            rejected_trajectories=noisy_trajs,
        )
        test_dataset = EmbodiedVLADataset(
            trajectories=test_trajs,
            chunk_size=cfg.chunk_size,
            is_train=False,
        )
    else:
        train_file = os.path.join(cfg.output_dir, "train_trajectories.npy")
        test_file = os.path.join(cfg.output_dir, "test_trajectories.npy")
        noisy_file = os.path.join(cfg.output_dir, "noisy_preference_trajectories.npy")
        manifest_file = os.path.join(cfg.output_dir, "multi_dataset_manifest.json")

        if not os.path.exists(train_file):
            print(f"[Pipeline] Processed dataset not found in '{cfg.output_dir}'. Auto-running MultiOpenXDataProcessor (Top-30 Shards)...")
            processor = MultiOpenXDataProcessor(base_dir=base_dir, output_dir=cfg.output_dir, max_shards_per_dataset=30)
            processor.process_all_datasets()

        train_trajs = np.load(train_file, allow_pickle=True).tolist()
        test_trajs = np.load(test_file, allow_pickle=True).tolist()
        noisy_trajs = np.load(noisy_file, allow_pickle=True).tolist() if os.path.exists(noisy_file) else None

        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_info = json.load(f)
                print(f"   📊 Multi-Embodiment Manifest Loaded: {len(manifest_info.get('datasets', {}))} Datasets Unified.")

        train_dataset = EmbodiedVLADataset(train_trajs, chunk_size=cfg.chunk_size, is_train=True, rejected_trajectories=noisy_trajs)
        test_dataset = EmbodiedVLADataset(test_trajs, chunk_size=cfg.chunk_size, is_train=False)

    print(f"   Dataset Split: {len(train_dataset)} Train Chunk Batches, {len(test_dataset)} Holdout Test Chunk Batches.")

    # 4. Model Construction & Checkpoint Loading
    print("\n[Step 2/5] 🧠 Initializing OpenVLAAlignFlow Architecture...")
    device = torch.device(cfg.device)
    model = OpenVLAAlignFlow(cfg).to(device)

    # Determine Resume Checkpoint
    ckpt_to_load = resume_path
    if ckpt_to_load is None and stage_str in ("2", "3", "eval", "eval_only"):
        # Auto-find matching checkpoint
        req_stage = 2 if stage_str == "2" else (3 if stage_str == "3" else "eval")
        ckpt_to_load = find_latest_checkpoint(checkpoint_dir, target_stage=req_stage)

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"🔄 [Breakpoint Resume] Loading weights from: {ckpt_to_load}")
        model.load_checkpoint(ckpt_to_load, map_location=cfg.device)
    elif stage_str in ("2", "3", "eval", "eval_only"):
        print(f"⚠️ [Warning] No preceding checkpoint found for start_stage='{stage_str}'. Initializing from current weights.")

    # 5. Stage 1: Fine-Grained VL Alignment
    if stage_str in ("1", "stage1", "all"):
        print("\n[Step 3/5] 🎯 Executing Stage 1: Sub-goal InfoNCE + Affordance Mask Alignment...")
        model = run_stage1_vl_alignment(
            model=model,
            config=cfg,
            custom_dataset=train_dataset,
            epochs=cfg.stage1_epochs,
        )
        stage1_path = os.path.join(run_ckpt_dir, "stage1_checkpoint.pt")
        model.save_checkpoint(stage1_path)
        print(f"💾 [Checkpoint] Stage 1 checkpoint saved to: {stage1_path}")
    else:
        print(f"\n⏩ [Skipping Stage 1] Resuming directly from Stage {stage_str.upper()}...")

    # 6. Stage 2: Conditional Flow Matching Action Head Pretraining
    if stage_str in ("1", "2", "stage1", "stage2", "all"):
        print("\n[Step 4/5] 🌊 Executing Stage 2: Optimal Transport CFM Action Head Pretraining...")
        model = run_stage2_flow_pretraining(
            model=model,
            config=cfg,
            custom_dataset=train_dataset,
            epochs=cfg.stage2_epochs,
        )
        stage2_path = os.path.join(run_ckpt_dir, "stage2_checkpoint.pt")
        model.save_checkpoint(stage2_path)
        print(f"💾 [Checkpoint] Stage 2 checkpoint saved to: {stage2_path}")
    else:
        print(f"\n⏩ [Skipping Stage 2] Resuming directly from Stage {stage_str.upper()}...")

    # 7. Stage 3: SOTA Continuous-Flow Trajectory DPO
    if stage_str in ("1", "2", "3", "stage1", "stage2", "stage3", "all"):
        print("\n[Step 5/5] ⚖️ Executing Stage 3: Offline Policy Trajectory-DPO Alignment...")
        model = run_stage3_offline_rl_dpo(
            model=model,
            config=cfg,
            custom_dataset=train_dataset,
            epochs=cfg.stage3_epochs,
        )
        stage3_path = os.path.join(run_ckpt_dir, "stage3_checkpoint.pt")
        model.save_checkpoint(stage3_path)
        print(f"💾 [Checkpoint] Stage 3 checkpoint saved to: {stage3_path}")
    else:
        print(f"\n⏩ [Skipping Stage 3] Direct Evaluation Mode...")

    # 8. Save Final Models
    final_ckpt_path = os.path.join(run_ckpt_dir, "openvla_alignflow_final.pt")
    model.save_checkpoint(final_ckpt_path)

    # Also update global master reference checkpoint in root ./checkpoints
    root_master_ckpt = os.path.join(checkpoint_dir, "openvla_alignflow_final.pt")
    if os.path.abspath(final_ckpt_path) != os.path.abspath(root_master_ckpt):
        model.save_checkpoint(root_master_ckpt)

    # 9. Offline Benchmark Evaluation
    print("\n📊 Running Multi-Dimensional Offline Benchmark Evaluation...")
    evaluator = OfflineBenchmarkEvaluator(model=model, device=device, dt=cfg.delta_t)
    metrics = evaluator.evaluate_dataset(
        test_dataset=test_dataset,
        num_samples=min(cfg.eval_num_samples, len(test_dataset)),
        ode_steps=cfg.ode_eval_steps,
    )

    report_json_path = os.path.join(run_ckpt_dir, "benchmark_report.json")
    evaluator.print_benchmark_report(metrics, save_path=report_json_path)

    print(f"🎉 OpenVLA-AlignFlow Execution Finished! All artifacts saved in: {run_ckpt_dir}\n")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Run OpenVLA-AlignFlow Pipeline (RTX 4090 High-Performance Edition)")
    parser.add_argument("--mode", type=str, default="full", choices=["demo", "full"], help="Execution mode (default: full)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Hardware device (default: cuda)")
    parser.add_argument("--stage1_epochs", type=int, default=None, help="Stage 1 Alignment Epochs (default: from config)")
    parser.add_argument("--stage2_epochs", type=int, default=None, help="Stage 2 CFM Epochs (default: from config)")
    parser.add_argument("--stage3_epochs", type=int, default=None, help="Stage 3 DPO Epochs (default: from config)")
    parser.add_argument("--start_stage", type=str, default="1", choices=["1", "2", "3", "eval", "eval_only"], help="Stage to start/resume from: 1, 2, 3, or eval (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from the latest available checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Explicit path to checkpoint .pt file to resume from")
    parser.add_argument("--exp_name", type=str, default=None, help="Custom experiment directory name (default: auto exp_YYYYMMDD_HHMMSS)")
    parser.add_argument("--base_dir", type=str, default=".", help="Root base directory")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Data output directory (default: ./data/processed)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoints root directory (default: ./checkpoints)")
    args = parser.parse_args()

    # If --resume flag is passed without explicit resume_path, auto-detect
    resume_path = args.resume_path
    if args.resume and resume_path is None:
        resume_path = find_latest_checkpoint(args.checkpoint_dir)
        if resume_path:
            print(f"🔍 [Auto-Resume] Found latest checkpoint: {resume_path}")

    run_pipeline(
        mode=args.mode,
        device_name=args.device,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        exp_name=args.exp_name,
        start_stage=args.start_stage,
        resume_path=resume_path,
    )


if __name__ == "__main__":
    main()
