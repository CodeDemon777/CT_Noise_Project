"""
Model 2 Visualization Module - Attention U-Net
Poisson = Red (0, 0, 255 in BGR)
Speckle = Green (0, 255, 0 in BGR)
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


# Color constants (BGR for OpenCV)
POISSON_COLOR_BGR = (0, 0, 255)    # Red
SPECKLE_COLOR_BGR = (0, 255, 0)    # Green
CLEAN_COLOR_BGR = (200, 200, 200)  # Light grey (for mask display)


class Model2Visualizer:
    """
    Generates pixel-level segmentation mask, colored overlay, and annotated CT
    image for Model 2 (Poisson + Speckle noise).
    """

    def __init__(self, original_image: np.ndarray):
        """
        Args:
            original_image: Grayscale numpy image (H, W) or (H, W, C).
        """
        if len(original_image.shape) == 2:
            self.original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.original_bgr = original_image.copy()

    def generate_colored_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        """
        Generate a color-coded segmentation mask image.
        0=Clean → dark  1=Poisson → Red  2=Speckle → Green
        """
        h, w = pred_mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[pred_mask == 0] = [20, 20, 20]            # dark background for clean
        colored_mask[pred_mask == 1] = POISSON_COLOR_BGR        # Red
        colored_mask[pred_mask == 2] = SPECKLE_COLOR_BGR        # Green
        return colored_mask

    def generate_overlay(self, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Blend the color mask with the original CT image for a transparent overlay.
        """
        colored_mask = self.generate_colored_mask(pred_mask)
        # Only blend where there is noise (non-zero)
        noise_pixels = pred_mask > 0
        overlay = self.original_bgr.copy()
        overlay[noise_pixels] = cv2.addWeighted(
            self.original_bgr, 1 - alpha, colored_mask, alpha, 0
        )[noise_pixels]
        return overlay

    def generate_annotated(self, pred_mask: np.ndarray, severity_report: Dict) -> np.ndarray:
        """
        Draw bounding boxes around contiguous noise regions on the CT image.
        Poisson = Red box, Speckle = Green box.
        Pixel segmentation is NOT replaced — boxes are additive on top of the overlay.
        """
        annotated = self.generate_overlay(pred_mask, alpha=0.35)

        for class_id, label, color in [
            (1, "Poisson", POISSON_COLOR_BGR),
            (2, "Speckle", SPECKLE_COLOR_BGR),
        ]:
            binary_mask = (pred_mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pct_val = severity_report["noise"]["poisson"]["severity_percentage"] if class_id == 1 \
                else severity_report["noise"]["speckle"]["severity_percentage"]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated,
                    f"{label} {pct_val:.1f}%",
                    (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA
                )

        return annotated

    def generate_full_visualization(
        self,
        pred_mask: np.ndarray,
        severity_report: Dict,
        output_path: str,
    ) -> Dict[str, np.ndarray]:
        """
        Generate all Model 2 visualizations and save them to disk.

        Saves:
          - <output_path> → annotated image
          - <stem>_mask.png → colored pixel mask
          - <stem>_overlay.png → blended overlay

        Returns:
            Dict with keys: 'mask', 'overlay', 'annotated'.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_path = out_path.parent / f"{out_path.stem}_mask.png"
        overlay_path = out_path.parent / f"{out_path.stem}_overlay.png"

        colored_mask = self.generate_colored_mask(pred_mask)
        overlay = self.generate_overlay(pred_mask, alpha=0.45)
        annotated = self.generate_annotated(pred_mask, severity_report)

        cv2.imwrite(str(mask_path), colored_mask)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(out_path), annotated)

        print(f"[Model 2] Saved mask to {mask_path}")
        print(f"[Model 2] Saved overlay to {overlay_path}")
        print(f"[Model 2] Saved annotated to {out_path}")

        return {
            "mask": colored_mask,
            "overlay": overlay,
            "annotated": annotated,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "annotated_path": str(out_path),
        }
