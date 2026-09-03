from pathlib import Path
import torch
from .model import NoiseCNN


def load_model4(model_path: str, device: str = "cpu") -> NoiseCNN:
    """
    Load Model 4 (NoiseCNN — Vasanth) checkpoint.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model 4 checkpoint not found: {path}")

    model = NoiseCNN(num_classes=3)
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"[OK] Model 4 (NoiseCNN) loaded successfully from {path} on {device}")
    return model
