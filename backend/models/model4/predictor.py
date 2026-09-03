from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .model_loader import load_model4


class Model4Predictor:
    """
    Inference wrapper for Model 4 (NoiseCNN).
    Preprocesses CT image to 128x128 with min-max normalization.
    Returns class probabilities for Clean, Quantization Noise, and Periodic Noise.
    """

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = load_model4(model_path, device=self.device)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Min-max normalization as trained
        img_f = image.astype(np.float32)
        denom = (img_f.max() - img_f.min()) + 1e-8
        norm = (img_f - img_f.min()) / denom

        resized = cv2.resize(norm, (128, 128), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_input) -> dict:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input must be a file path string/Path or numpy.ndarray")

        tensor = self.preprocess(img)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return {
            "clean": float(probs[0]),
            "quantization": float(probs[1]),
            "periodic": float(probs[2]),
        }
