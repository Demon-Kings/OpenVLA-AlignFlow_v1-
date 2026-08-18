"""
Open X-Embodiment (OpenX) Dataset Multi-Source Downloader
Supports: BridgeData v2, Fractal20220817 (RT-1), DROID-100
"""
import os
import sys
import argparse
import subprocess
from typing import Dict, Any, List, Optional


OPENX_DATASET_REGISTRY = {
    "bridge_v2": {
        "name": "BridgeData v2 (WidowX Manipulator)",
        "gcs_path": "gs://gresearch/robotics/bridge_dataset/0.1.0",
        "hf_repo": "rail-berkeley/bridge_dataset",
        "approx_size_full": "~60 GB",
        "mini_shards": ["bridge_dataset-train.tfrecord-00000-of-01024", "bridge_dataset-train.tfrecord-00001-of-01024", "bridge_dataset-train.tfrecord-00002-of-01024", "bridge_dataset-train.tfrecord-00003-of-01024", "bridge_dataset-train.tfrecord-00004-of-01024"],
        "metadata_files": ["dataset_info.json", "features.json"],
    },
    "fractal_rt1": {
        "name": "Fractal20220817 / RT-1 (Google Robot)",
        "gcs_path": "gs://gresearch/robotics/fractal20220817_data/0.1.0",
        "hf_repo": "google/fractal20220817_data",
        "approx_size_full": "~300 GB",
        "mini_shards": ["fractal20220817_data-train.tfrecord-00000-of-01024", "fractal20220817_data-train.tfrecord-00001-of-01024"],
        "metadata_files": ["dataset_info.json", "features.json"],
    },
    "droid_100": {
        "name": "DROID 100 (Franka Panda 7-DoF Multi-Scene)",
        "gcs_path": "gs://gresearch/robotics/droid_100",
        "hf_repo": "droid-dataset/droid",
        "approx_size_full": "~40 GB",
        "mini_shards": ["1.0.0/r2d2_faceblur-train.tfrecord-00000-of-00031", "1.0.0/r2d2_faceblur-train.tfrecord-00001-of-00031"],
        "metadata_files": ["dataset_info.json", "features.json"],
    },
}


class OpenXDatasetDownloader:
    def __init__(self, target_dir: str = "./data/openx", hf_endpoint: str = "https://hf-mirror.com"):
        self.target_dir = target_dir
        self.hf_endpoint = hf_endpoint
        os.makedirs(self.target_dir, exist_ok=True)

    def print_available_datasets(self):
        print("\n" + "=" * 80)
        print("📦 Open X-Embodiment Available Dataset Registry (3 Core Sets):")
        print("=" * 80)
        for key, meta in OPENX_DATASET_REGISTRY.items():
            print(f"• [{key:<15s}] {meta['name']}")
            print(f"   - GCS Path  : {meta['gcs_path']}")
            print(f"   - Full Size : {meta['approx_size_full']}")
            print(f"   - HF Mirror : {self.hf_endpoint}/{meta['hf_repo']}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download Open X-Embodiment Datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bridge_v2",
        choices=list(OPENX_DATASET_REGISTRY.keys()) + ["all", "list"],
        help="Dataset key to download",
    )
    args = parser.parse_args()

    downloader = OpenXDatasetDownloader()
    downloader.print_available_datasets()


if __name__ == "__main__":
    main()
