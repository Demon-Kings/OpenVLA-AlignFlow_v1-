"""
Top-level entrypoint launcher for OpenVLA-AlignFlow
Directly executes with all production defaults in code:
  --mode full
  --device cuda (with automatic fallback to cpu if unavailable)
  --output_dir ./data/processed
  --stage1_epochs 15
  --stage2_epochs 25
  --stage3_epochs 10
"""
import sys
import os

# Add directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla.run_pipeline import main, run_pipeline

if __name__ == "__main__":
    main()
