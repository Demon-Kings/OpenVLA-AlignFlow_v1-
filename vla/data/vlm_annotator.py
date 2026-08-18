"""
VLM Semantic Instruction Expansion & Sub-Goal Keyframe Anchor Annotator
"""
import numpy as np
from typing import List, Dict, Any, Optional


class VLMAnnotator:
    """
    Simulates / integrates VLM annotation pipeline:
    1. Multi-perspective language instruction paraphrasing (5x spatial expansion).
    2. Sub-goal critical frame anchor extraction based on gripper state transition & kinetic extrema.
    """

    TEMPLATES = {
        "pick": [
            "pick up the {obj}",
            "grasp the {obj} and lift it up",
            "reach out and grab the {obj}",
            "retrieve the {obj} from the tabletop",
            "securely hold the {obj} with gripper",
        ],
        "place": [
            "place the object into the {target}",
            "put down the item on the {target}",
            "transfer and set the item inside the {target}",
            "release gripper over the {target}",
            "deposit the carried object into the {target}",
        ],
        "wipe": [
            "wipe the {target} with sponge",
            "clean the surface of the {target}",
            "slide across the {target} back and forth",
            "sweep the tabletop surface of {target}",
            "execute wiping motion over {target}",
        ],
        "open": [
            "open the {target}",
            "pull the handle of {target}",
            "grasp the door knob and pull open {target}",
            "swing open the {target}",
            "slide the drawer of {target} open",
        ],
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def expand_instruction(self, base_instruction: str, num_expansions: int = 5) -> List[str]:
        """
        Generate 5 diverse, spatially-grounded paraphrased instructions for a task.
        """
        base_lower = base_instruction.lower().strip()

        matched_key = None
        for key in self.TEMPLATES:
            if key in base_lower:
                matched_key = key
                break

        if matched_key is None:
            matched_key = "pick"

        # Extract target or object noun phrase
        words = base_lower.split()
        target = words[-1] if len(words) > 1 else "target"
        obj = words[-1] if len(words) > 1 else "object"

        expanded = []
        for tmpl in self.TEMPLATES[matched_key][:num_expansions]:
            text = tmpl.format(obj=obj, target=target)
            expanded.append(text)

        while len(expanded) < num_expansions:
            expanded.append(base_instruction)

        return expanded

    def extract_subgoal_keyframes(
        self,
        actions: np.ndarray,
        gripper_channel: int = 6,
        velocity_window: int = 3,
    ) -> List[int]:
        """
        Detect sub-goal milestone frame indices based on gripper actuation points
        and kinetic deceleration phases.
        """
        actions = np.asarray(actions, dtype=np.float32)
        L = len(actions)
        if L < 5:
            return [L - 1]

        subgoals = set()

        # 1. Gripper transition triggers (gripper state changed from open to closed or vice versa)
        if actions.shape[1] > gripper_channel:
            gripper_signal = actions[:, gripper_channel]
            gripper_diff = np.abs(np.diff(gripper_signal))
            change_indices = np.where(gripper_diff > 0.3)[0]
            for idx in change_indices:
                subgoals.add(int(idx))

        # 2. Minimum velocity inflection points (approaching object or goal before direction change)
        if actions.shape[1] >= 3:
            vels = np.linalg.norm(actions[:, :3], axis=1)
            for i in range(velocity_window, L - velocity_window):
                if vels[i] < np.min(vels[i - velocity_window : i]) and vels[i] < np.min(vels[i + 1 : i + velocity_window + 1]):
                    subgoals.add(i)

        # 3. Always include terminal milestone frame
        subgoals.add(L - 1)

        sorted_subgoals = sorted(list(subgoals))
        return sorted_subgoals
