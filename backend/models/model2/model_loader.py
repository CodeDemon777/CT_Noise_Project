"""
Model Loader for Model 2 (Attention U-Net)
"""

import sys
from pathlib import Path
import gc
import torch

from .model import AttentionUNet


def load_model2(model_path: str, device: str = None) -> torch.nn.Module:
    """
    Load the Attention U-Net model checkpoint with ultra-low RAM footprint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_device = torch.device(device)
    model2 = AttentionUNet(in_channels=1, out_channels=4)

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model 2 checkpoint not found at: {model_path}")

    # Load checkpoint safely
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        checkpoint = torch.load(str(path), map_location="cpu")

    # Unwrap state_dict if contained inside dictionary wrapper
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    del checkpoint
    gc.collect()

    model2.load_state_dict(state_dict)
    del state_dict
    gc.collect()

    model2.to(target_device)
    model2.eval()

    print(f"[OK] Model 2 (Attention U-Net) loaded successfully from {path} on {device}")
    return model2
