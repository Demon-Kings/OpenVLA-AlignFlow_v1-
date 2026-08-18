"""
BridgeData v2 Dataset Adapter (WidowX Manipulator)
"""
import numpy as np
from typing import Dict, Any, Optional, List
from .base_adapter import BaseEmbodimentAdapter


class BridgeAdapter(BaseEmbodimentAdapter):
    """
    Adapter for BridgeData v2:
      - Embodiment: WidowX 250s (ID 0)
      - Observation Image Key: 'observation/image_0' or 'image_0'
      - Action: 7-DoF EEF Delta Pose [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """

    def __init__(self, image_size: int = 224):
        super().__init__(embodiment_key="widowx", embodiment_id=0, image_size=image_size)

    def get_dataset_key(self) -> str:
        return "bridge_dataset"

    def parse_episode(self, raw_episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images = raw_episode.get("images", raw_episode.get("observation/image_0"))
        actions = raw_episode.get("actions", raw_episode.get("action"))
        instructions = raw_episode.get("instructions", raw_episode.get("language_instruction", ["wipe the table"]))

        if images is None or actions is None:
            return None

        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim != 2 or actions_arr.shape[1] < 7:
            return None

        # Format instructions
        if isinstance(instructions, (str, bytes)):
            text = instructions.decode("utf-8") if isinstance(instructions, bytes) else str(instructions)
            inst_list = [text]
        else:
            inst_list = [str(x) for x in instructions]

        return {
            "traj_id": raw_episode.get("traj_id", "bridge_0000"),
            "embodiment": self.embodiment_key,
            "embodiment_id": self.embodiment_id,
            "images": images,
            "actions": actions_arr[:, :7],
            "instructions": inst_list,
            "subgoals": raw_episode.get("subgoals", [len(actions_arr) - 1]),
            "is_expert": raw_episode.get("is_expert", True),
            "return": float(raw_episode.get("return", 1.0)),
        }
