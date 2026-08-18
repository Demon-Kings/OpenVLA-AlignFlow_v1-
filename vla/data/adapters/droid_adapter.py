"""
DROID-100 Dataset Adapter (Franka Emika Panda)
"""
import numpy as np
from typing import Dict, Any, Optional, List
from .base_adapter import BaseEmbodimentAdapter


class DroidAdapter(BaseEmbodimentAdapter):
    """
    Adapter for DROID / DROID-100:
      - Embodiment: Franka Emika Panda (ID 2)
      - Observation Image Key: 'observation/exterior_image_1_left' or 'image_0'
      - Action: 7-DoF Franka Cartesian EEF Velocity/Position + Gripper
    """

    def __init__(self, image_size: int = 224):
        super().__init__(embodiment_key="franka_panda", embodiment_id=2, image_size=image_size)

    def get_dataset_key(self) -> str:
        return "droid_100"

    def parse_episode(self, raw_episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images = raw_episode.get("images", raw_episode.get("observation/exterior_image_1_left", raw_episode.get("observation/image_0")))
        actions = raw_episode.get("actions", raw_episode.get("action/cartesian_velocity", raw_episode.get("action")))

        if images is None or actions is None:
            return None

        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim != 2 or actions_arr.shape[1] < 7:
            return None

        instructions = raw_episode.get("instructions", raw_episode.get("language_instruction", ["manipulate object carefully"]))
        if isinstance(instructions, (str, bytes)):
            text = instructions.decode("utf-8") if isinstance(instructions, bytes) else str(instructions)
            inst_list = [text]
        else:
            inst_list = [str(x) for x in instructions]

        return {
            "traj_id": raw_episode.get("traj_id", "droid_0000"),
            "embodiment": self.embodiment_key,
            "embodiment_id": self.embodiment_id,
            "images": images,
            "actions": actions_arr[:, :7],
            "instructions": inst_list,
            "subgoals": raw_episode.get("subgoals", [len(actions_arr) - 1]),
            "is_expert": raw_episode.get("is_expert", True),
            "return": float(raw_episode.get("return", 1.0)),
        }
