from pathlib import Path
import gc
import torch
from .model import NoiseCNN


def load_model4(model_path: str, device: str = "cpu") -> NoiseCNN:
    """
    Load Model 4 (NoiseCNN — Vasanth) checkpoint with zero-copy memory mapping.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model 4 checkpoint not found: {path}")

    model = NoiseCNN(num_classes=3)
    try:
        checkpoint = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = torch.load(str(path), map_location="cpu")

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

    print(f"[OK] Model 4 (NoiseCNN) loaded successfully from {path} on {device}")
    return model
