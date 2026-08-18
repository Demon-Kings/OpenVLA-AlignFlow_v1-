"""
Multi-Embodiment Dataset Adapters Package
"""
from .base_adapter import BaseEmbodimentAdapter
from .bridge_adapter import BridgeAdapter
from .fractal_adapter import FractalAdapter
from .droid_adapter import DroidAdapter

ADAPTER_REGISTRY = {
    "bridge_dataset": BridgeAdapter,
    "fractal20220817_data": FractalAdapter,
    "droid_100": DroidAdapter,
}

def get_adapter_for_dataset(dataset_key: str, image_size: int = 224) -> BaseEmbodimentAdapter:
    """Factory helper to instantiate the appropriate adapter for a dataset."""
    for key, adapter_cls in ADAPTER_REGISTRY.items():
        if key in dataset_key:
            return adapter_cls(image_size=image_size)
    return BridgeAdapter(image_size=image_size)

__all__ = [
    "BaseEmbodimentAdapter",
    "BridgeAdapter",
    "FractalAdapter",
    "DroidAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter_for_dataset",
]
