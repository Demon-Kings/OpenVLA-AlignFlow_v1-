"""
Prepare and Process Exactly the Top-30 Shard Files (TFRecord Slices) Across All 3 OpenX Datasets
Combines:
  - BridgeData v2 (WidowX): Top-30 Shards (00000 to 00029)
  - Fractal20220817 (RT-1): Top-30 Shards (00000 to 00029)
  - DROID-100 (Franka): Top-30 Shards (00000 to 00029)
Total: 90 Shard Files -> ./data/processed/
"""
import os
import sys

# Ensure package root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla.data.process_multi_openx import MultiOpenXDataProcessor


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data", "processed")
    
    print(f"Executing Multi-Embodiment Data Preparation across 3 datasets (Top-30 Shards per dataset)...")
    print(f"Base Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")

    # Process exactly top-30 shard files (TFRecord slices) per dataset
    processor = MultiOpenXDataProcessor(
        base_dir=base_dir,
        output_dir=output_dir,
        max_shards_per_dataset=50,
    )
    processor.process_all_datasets()


if __name__ == "__main__":
    main()
