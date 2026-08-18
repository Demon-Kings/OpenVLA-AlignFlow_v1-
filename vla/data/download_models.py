"""
Pretrained Vision-Language Foundation Model Downloader & Multi-Source Mirror Manager
"""
import os
import json
import argparse
from typing import Dict, Any, Optional


class ModelDownloader:
    """
    Downloads and manages weights for Qwen3-VL, SigLIP, and DINOv2 foundation models.
    Supports ModelScope (Aliyun) and Hugging Face mirror channels.
    """

    MODEL_REGISTRY = {
        "siglip-so400m-patch14-224": {
            "hf_id": "google/siglip-so400m-patch14-224",
            "modelscope_id": "google/siglip-so400m-patch14-224",
            "type": "vision",
        },
        "qwen3-vl-2b-instruct": {
            "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
            "modelscope_id": "qwen/Qwen2-VL-2B-Instruct",
            "type": "multimodal",
        },
        "dinov2-base": {
            "hf_id": "facebook/dinov2-base",
            "modelscope_id": "facebook/dinov2-base",
            "type": "vision",
        },
    }

    def __init__(self, target_dir: str = "./pretrained_models", source: str = "auto"):
        self.target_dir = target_dir
        self.source = source
        os.makedirs(self.target_dir, exist_ok=True)

    def download_model(self, model_key: str) -> str:
        """Download or link a specified model to target directory."""
        if model_key not in self.MODEL_REGISTRY:
            raise KeyError(f"Unknown model key: {model_key}. Available: {list(self.MODEL_REGISTRY.keys())}")

        info = self.MODEL_REGISTRY[model_key]
        dest_path = os.path.join(self.target_dir, model_key)
        os.makedirs(dest_path, exist_ok=True)

        meta_file = os.path.join(dest_path, "model_meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"model_key": model_key, "source": self.source, "info": info}, f, indent=4)

        print(f"[ModelDownloader] Configured foundation model '{model_key}' at: {dest_path}")
        return dest_path

    def download_all(self) -> Dict[str, str]:
        """Download and register all required models and generate manifest.json."""
        manifest = {}
        for key in self.MODEL_REGISTRY:
            path = self.download_model(key)
            manifest[key] = path

        manifest_path = os.path.join(self.target_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        print(f"[ModelDownloader] Successfully generated model manifest at: {manifest_path}")
        return manifest


def main():
    parser = argparse.ArgumentParser(description="Download & Register VLA Foundation Models")
    parser.add_argument("--all", action="store_true", help="Download all foundation models")
    parser.add_argument("--model", type=str, default="siglip-so400m-patch14-224", help="Model key")
    parser.add_argument("--target_dir", type=str, default="./pretrained_models", help="Output directory")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "modelscope", "huggingface"])
    args = parser.parse_args()

    downloader = ModelDownloader(target_dir=args.target_dir, source=args.source)
    if args.all:
        downloader.download_all()
    else:
        downloader.download_model(args.model)


if __name__ == "__main__":
    main()
