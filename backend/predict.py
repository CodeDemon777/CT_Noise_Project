"""
Prediction Module
Handles image preprocessing, model inference, and postprocessing
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
from .model_loader import load_model


class CTPredictor:
    """
    Handles prediction pipeline for CT noise detection
    """
    
    # Class definitions
    CLASS_NAMES = {
        0: "Clean",
        1: "Gaussian",
        2: "Poisson"
    }
    
    CLASS_COLORS = {
        0: (0, 255, 0),      # Green
        1: (0, 0, 255),      # Red (Gaussian)
        2: (255, 0, 0),      # Blue (Poisson)
    }
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize predictor with model
        
        Args:
            model_path: Path to best_model.pth
            device: Device to run inference on
        """
        self.model = load_model(model_path, device)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def read_ct_image(self, image_path: str) -> np.ndarray:
        """
        Read CT image from file
        
        Args:
            image_path: Path to CT image (DICOM, PNG, JPG, etc.)
        
        Returns:
            Image as numpy array (grayscale, 0-255)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        print(f"✅ Loaded image: {image_path.name} | Shape: {img.shape}")
        return img
    
    def preprocess(self, image: np.ndarray, target_size: int = 512) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model inference
        
        Args:
            image: Input CT image
            target_size: Target size for model input
        
        Returns:
            Preprocessed tensor and original shape
        """
        original_shape = image.shape
        
        # Resize to target size
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [-1, 1]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch and channel dimensions
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        print(f"📊 Preprocessed shape: {tensor.shape}")
        return tensor, original_shape
    
    def predict_mask(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Run model inference
        
        Args:
            image_tensor: Preprocessed image tensor
        
        Returns:
            Predicted mask (H, W) with class indices
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Get class with highest probability
            predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        print(f"🔍 Inference complete | Output shape: {predicted.shape}")
        return predicted
    
    def postprocess(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess predicted mask to original size
        
        Args:
            mask: Predicted mask
            original_shape: Original image shape
        
        Returns:
            Resized mask matching original image
        """
        # Resize back to original shape
        postprocessed = cv2.resize(
            mask.astype(np.uint8),
            original_shape[::-1],
            interpolation=cv2.INTER_NEAREST
        )
        
        print(f"✅ Postprocessed mask shape: {postprocessed.shape}")
        return postprocessed
    
    def predict(self, image_path: str) -> np.ndarray:
        """
        Full prediction pipeline
        
        Args:
            image_path: Path to input CT image
        
        Returns:
            Predicted segmentation mask
        """
        # Read
        image = self.read_ct_image(image_path)
        
        # Preprocess
        tensor, orig_shape = self.preprocess(image)
        
        # Predict
        mask = self.predict_mask(tensor)
        
        # Postprocess
        final_mask = self.postprocess(mask, orig_shape)
        
        return final_mask


if __name__ == "__main__":
    # Test prediction
    model_path = Path(__file__).parent.parent / "model" / "best_model.pth"
    predictor = CTPredictor(str(model_path))
    
    # Create a dummy test image
    test_image_path = Path(__file__).parent.parent / "test_ct.png"
    if not test_image_path.exists():
        print("📝 Creating test image...")
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        cv2.imwrite(str(test_image_path), test_image)
    
    # Run prediction
    mask = predictor.predict(str(test_image_path))
    print(f"Prediction shape: {mask.shape}")
    print(f"Unique classes: {np.unique(mask)}")
