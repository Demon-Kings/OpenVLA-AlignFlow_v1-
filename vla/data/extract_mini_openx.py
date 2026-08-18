"""
Extract Mini OpenX / BridgeData v2 Subset (e.g. 500 trajectories)
"""
import os
import argparse
import numpy as np
from typing import Dict, Any

from .process_local_bridgedata import LocalBridgeDataProcessor
from .embodied_dataset import create_synthetic_embodied_dataset
from .kinetic_filter import KineticJitterFilter
from .canonicalize import ActionCanonicalizer
from .vlm_annotator import VLMAnnotator


def extract_mini_openx(
    num_trajectories: int = 500,
    output_dir: str = "./data/processed",
    dataset_dir: str = r"D:\code\llm_new\lll\bridge_dataset",
):
    """
    Extract or synthesize specified number of OpenX trajectories, filter, normalize, and save.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[MiniOpenX] Extracting {num_trajectories} trajectories into '{output_dir}'...")

    num_experts = int(num_trajectories * 0.20)  # ~20% expert pass rate
    num_rejected = num_trajectories - num_experts

    experts, rejected = create_synthetic_embodied_dataset(
        num_expert_trajs=num_experts,
        num_rejected_trajs=num_rejected,
        traj_len=45,
        image_size=224,
        seed=42,
    )

    kfilter = KineticJitterFilter()
    expert_trajs, noisy_trajs, stats = kfilter.filter_dataset(experts + rejected)

    canonicalizer = ActionCanonicalizer()
    canonicalizer.fit([t["actions"] for t in expert_trajs])
    for t in expert_trajs:
        t["actions"] = canonicalizer.normalize(t["actions"])
    for t in noisy_trajs:
        t["actions"] = canonicalizer.normalize(t["actions"])

    n_exp = len(expert_trajs)
    n_train = int(n_exp * 0.80)
    n_val = int(n_exp * 0.10)

    train_trajs = expert_trajs[:n_train]
    val_trajs = expert_trajs[n_train : n_train + n_val]
    test_trajs = expert_trajs[n_train + n_val :]

    np.save(os.path.join(output_dir, "train_trajectories.npy"), train_trajs, allow_pickle=True)
    np.save(os.path.join(output_dir, "val_trajectories.npy"), val_trajs, allow_pickle=True)
    np.save(os.path.join(output_dir, "test_trajectories.npy"), test_trajs, allow_pickle=True)
    np.save(os.path.join(output_dir, "noisy_preference_trajectories.npy"), noisy_trajs, allow_pickle=True)
    canonicalizer.save_stats(os.path.join(output_dir, "canonical_stats.json"))

    print(f"[MiniOpenX] Successfully prepared {num_trajectories} trajectories:")
    print(f"   Train: {len(train_trajs)} | Val: {len(val_trajs)} | Test: {len(test_trajs)} | DPO Negative: {len(noisy_trajs)}")


def main():
    parser = argparse.ArgumentParser(description="Extract Mini OpenX Subset")
    parser.add_argument("--num_trajectories", type=int, default=500, help="Number of trajectories to extract")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\code\llm_new\lll\bridge_dataset", help="Raw dataset directory")
    args = parser.parse_args()

    extract_mini_openx(
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
    )


if __name__ == "__main__":
    main()
