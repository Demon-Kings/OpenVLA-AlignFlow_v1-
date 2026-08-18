"""
Fractal20220817 (RT-1) Dataset Adapter (Google Robot)
"""
import numpy as np
from typing import Dict, Any, Optional, List
from .base_adapter import BaseEmbodimentAdapter


class FractalAdapter(BaseEmbodimentAdapter):
    """
    Adapter for Fractal20220817 (RT-1):
      - Embodiment: Google Robot (ID 1)
      - Observation Image Key: 'observation/image' or 'image'
      - Action: world_vector (3) + rotation_delta (3) + gripper_closedness_action (1) -> 7-DoF
    """

    def __init__(self, image_size: int = 224):
        super().__init__(embodiment_key="google_robot", embodiment_id=1, image_size=image_size)

    def get_dataset_key(self) -> str:
        return "fractal20220817_data"

    def parse_episode(self, raw_episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images = raw_episode.get("images", raw_episode.get("observation/image"))
        actions = raw_episode.get("actions")

        # Handle RT-1 dictionary action structure if present
        if actions is None and "action/world_vector" in raw_episode:
            wv = raw_episode["action/world_vector"]
            rd = raw_episode.get("action/rotation_delta", np.zeros_like(wv))
            gc = raw_episode.get("action/gripper_closedness_action", np.ones((len(wv), 1)))
            actions = np.concatenate([wv, rd, gc], axis=-1)

        if images is None or actions is None:
            return None

        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim != 2 or actions_arr.shape[1] < 7:
            return None

        instructions = raw_episode.get("instructions", raw_episode.get("natural_language_instruction", ["open the drawer"]))
        if isinstance(instructions, (str, bytes)):
            text = instructions.decode("utf-8") if isinstance(instructions, bytes) else str(instructions)
            inst_list = [text]
        else:
            inst_list = [str(x) for x in instructions]

        return {
            "traj_id": raw_episode.get("traj_id", "fractal_0000"),
            "embodiment": self.embodiment_key,
            "embodiment_id": self.embodiment_id,
            "images": images,
            "actions": actions_arr[:, :7],
            "instructions": inst_list,
            "subgoals": raw_episode.get("subgoals", [len(actions_arr) - 1]),
            "is_expert": raw_episode.get("is_expert", True),
            "return": float(raw_episode.get("return", 1.0)),
        }
