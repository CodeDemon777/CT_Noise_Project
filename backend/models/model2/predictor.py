"""
Model 2 Predictor Module - Attention U-Net Inference Pipeline
"""

from pathlib import Path
import cv2
import numpy as np
import torch

from .model_loader import load_model2


class Model2Predictor:
    """
    Inference pipeline for Model 2 (Attention U-Net for Poisson + Speckle noise segmentation).
    """

    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model2(model_path, device=self.device)

    def read_ct_image(self, image_path: str) -> np.ndarray:
        """
        Read image and ensure 2D grayscale format.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read CT image from: {image_path}")
        return img

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for Model 2:
        - Grayscale
        - Resize to 512x512
        - Normalize by 255.0 to [0, 1]
        - Add batch & channel dims -> [1, 1, 512, 512]
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_path: str) -> np.ndarray:
        """
        Run complete prediction pipeline on an input image.

        Returns:
            pred_mask (np.ndarray): 2D integer mask with values 0 (Clean), 1 (Poisson), 2 (Speckle).
        """
        original_image = self.read_ct_image(image_path)
        original_height, original_width = original_image.shape[:2]

        input_tensor = self.preprocess(original_image)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        predictions = np.argmax(probs, axis=0).astype(np.uint8)

        # Standardized class mapping for Joshna.pth:
        # Raw outputs from Joshna.pth:
        # Channel 0: Clean / Background
        # Channel 1: Auxiliary Poisson
        # Channel 2: Speckle Noise
        # Channel 3: Primary Poisson Noise
        #
        # Target classes:
        # 0 = Clean, 1 = Poisson Noise, 2 = Speckle Noise
        mapped = np.zeros_like(predictions, dtype=np.uint8)
        mapped[predictions == 3] = 1  # Primary Poisson Noise
        mapped[predictions == 1] = 1  # Auxiliary Poisson Noise
        mapped[predictions == 2] = 2  # Speckle Noise

        # Multi-label co-occurrence sensitivity:
        # Detect Speckle noise wherever Channel 2 probability is significant (> 0.15)
        # ensuring both Speckle and Poisson noise are identified and visualized simultaneously.
        p_speckle = probs[2]
        p_poisson = np.maximum(probs[3], probs[1])
        speckle_detect = (p_speckle > 0.15) & (p_speckle >= p_poisson * 0.70) & (mapped == 0)
        mapped[speckle_detect] = 2

        # Resize mask back to original dimensions using INTER_NEAREST
        pred_mask = cv2.resize(
            mapped,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        return pred_mask
