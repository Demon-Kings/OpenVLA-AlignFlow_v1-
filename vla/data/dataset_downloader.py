"""
OpenX & BridgeData v2 Dataset ETL & Preprocessing Pipeline
"""
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from .kinetic_filter import KineticJitterFilter
from .canonicalize import ActionCanonicalizer
from .vlm_annotator import VLMAnnotator


class BridgeDatasetETL:
    """
    Extract, Transform, and Load (ETL) pipeline for robotic demonstration datasets.
    Parses raw episode structures, runs kinetic filtering, applies 7-DoF canonicalization,
    and partitions the dataset into Train, Val, Test, and DPO Negative pools.
    """

    def __init__(
        self,
        output_dir: str = "./data/processed",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ):
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.filter = KineticJitterFilter()
        self.canonicalizer = ActionCanonicalizer()
        self.annotator = VLMAnnotator()

        os.makedirs(self.output_dir, exist_ok=True)

    def process_and_save(self, raw_trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute full ETL pipeline on list of trajectory dictionaries.
        """
        print(f"[ETL] Total raw input trajectories: {len(raw_trajectories)}")

        # Step 1: Kinetic Jitter Filtering
        expert_trajs, noisy_trajs, stats = self.filter.filter_dataset(raw_trajectories)
        print(f"[ETL] Filter stats: {stats['expert_count']} experts, {stats['rejected_count']} rejected.")

        # Step 2: Canonicalize Actions
        expert_actions = [t["actions"] for t in expert_trajs]
        if expert_actions:
            self.canonicalizer.fit(expert_actions)
            for traj in expert_trajs:
                traj["actions"] = self.canonicalizer.normalize(traj["actions"])
            for traj in noisy_trajs:
                traj["actions"] = self.canonicalizer.normalize(traj["actions"])

        # Step 3: Sub-goal Keyframe Extraction & Instruction Expansion
        for traj in expert_trajs + noisy_trajs:
            if "subgoals" not in traj or not traj["subgoals"]:
                traj["subgoals"] = self.annotator.extract_subgoal_keyframes(traj["actions"])
            if "instructions" not in traj or not traj["instructions"]:
                traj["instructions"] = self.annotator.expand_instruction("pick up the object")

        # Step 4: Split Expert Trajectories
        num_exp = len(expert_trajs)
        n_train = int(num_exp * self.train_ratio)
        n_val = int(num_exp * self.val_ratio)

        train_trajs = expert_trajs[:n_train]
        val_trajs = expert_trajs[n_train : n_train + n_val]
        test_trajs = expert_trajs[n_train + n_val :]

        # Save to disk
        np.save(os.path.join(self.output_dir, "train_trajectories.npy"), train_trajs, allow_pickle=True)
        np.save(os.path.join(self.output_dir, "val_trajectories.npy"), val_trajs, allow_pickle=True)
        np.save(os.path.join(self.output_dir, "test_trajectories.npy"), test_trajs, allow_pickle=True)
        np.save(os.path.join(self.output_dir, "noisy_preference_trajectories.npy"), noisy_trajs, allow_pickle=True)
        self.canonicalizer.save_stats(os.path.join(self.output_dir, "canonical_stats.json"))

        summary = {
            "num_total": len(raw_trajectories),
            "num_train": len(train_trajs),
            "num_val": len(val_trajs),
            "num_test": len(test_trajs),
            "num_dpo_rejected": len(noisy_trajs),
            "filter_stats": stats,
        }
        print(f"[ETL] ETL processing complete. Saved dataset files to '{self.output_dir}'")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Dataset ETL Preprocessing")
    parser.add_argument("--data_root", type=str, default="./data/openx", help="Raw dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Processed output directory")
    args = parser.parse_args()

    etl = BridgeDatasetETL(output_dir=args.output_dir)
    print(f"[ETL] Ready to process datasets from {args.data_root}")


if __name__ == "__main__":
    main()
