from pathlib import Path
import gc
import torch
from .model import DeepLabV3Plus


def load_model3(model_path: str, device: str = "cpu") -> DeepLabV3Plus:
    """
    Load Model 3 (DeepLabV3+ — Jahnavi) checkpoint with ultra-low RAM footprint.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model 3 checkpoint not found: {path}")

    model = DeepLabV3Plus(num_classes=3)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    del checkpoint
    gc.collect()

    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()

    model.to(device)
    model.eval()

    print(f"[OK] Model 3 (DeepLabV3+) loaded successfully from {path} on {device}")
    return model
