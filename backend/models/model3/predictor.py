from pathlib import Path
import cv2
import numpy as np
import torch
from .model_loader import load_model3


class Model3Predictor:
    """
    Inference wrapper for Model 3 (DeepLabV3+).
    Input: grayscale or RGB image (resized to 512x512, normalized [0, 1])
    Output: 2D integer mask (512x512) with values 0 (clean), 1 (Salt & Pepper), 2 (RVIN).
    """

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = load_model3(model_path, device=self.device)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_input) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input must be a file path string/Path or numpy.ndarray")

        orig_h, orig_w = img.shape[:2]
        tensor = self.preprocess(img)

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if (orig_w, orig_h) != (512, 512):
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return pred
