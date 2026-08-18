"""
Abstract Base Class for Multi-Embodiment OpenX Dataset Adapters
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class BaseEmbodimentAdapter(ABC):
    """
    Abstract adapter for transforming raw dataset episodes into standardized format:
      - RGB Images: (L, 224, 224, 3) uint8
      - Canonical Actions: (L, 7) float32 [dx, dy, dz, droll, dpitch, dyaw, gripper]
      - Instructions: List[str]
      - Sub-goals: List[int]
      - Embodiment metadata
    """

    def __init__(self, embodiment_key: str, embodiment_id: int, image_size: int = 224):
        self.embodiment_key = embodiment_key
        self.embodiment_id = embodiment_id
        self.image_size = image_size

    @abstractmethod
    def parse_episode(self, raw_episode: Any) -> Optional[Dict[str, Any]]:
        """Parse a single raw episode into the standardized dictionary format."""
        pass

    @abstractmethod
    def get_dataset_key(self) -> str:
        """Return the unique dataset identifier key."""
        pass
