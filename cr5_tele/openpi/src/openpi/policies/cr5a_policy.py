"""CR5A Dobot policy — transforms between LeRobot dataset format and pi0 model format.

Dataset keys:
  - observation.state           → [j1..j6, gripper]  (7D)
  - action                      → [dx,dy,dz,dRx,dRy,dRz,gripper]  (7D, already deltas)
  - observation.images.d415     → wrist camera
  - observation.images.d435     → base / 3rd-person camera
  - prompt                      → task instruction (optional)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_cr5a_example() -> dict:
    """Creates a random input example for the CR5A policy (used for shape inference)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CR5AInputs(transforms.DataTransformFn):
    """Convert CR5A dataset observation to pi0 model input format.

    Maps:
      - observation.image      → base_0_rgb      (D435 scene camera)
      - observation.wrist_image → left_wrist_0_rgb (D415 wrist camera)
      - observation.state      → state            (7D: j1..j6 + gripper)
      - actions                → actions          (7D delta + gripper)
      - prompt                 → prompt           (task instruction)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CR5AOutputs(transforms.DataTransformFn):
    """Extract the first 7 action dims from the model output (CR5A has 7D action)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
