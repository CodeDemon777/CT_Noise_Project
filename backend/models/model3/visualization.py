from pathlib import Path
import cv2
import numpy as np


class Model3Visualizer:
    """
    Visualizer for Model 3 (DeepLabV3+):
      Class 1 = Salt & Pepper (Orange: BGR 0, 140, 255)
      Class 2 = RVIN (Purple: BGR 211, 0, 148)
    """

    COLOR_SALT_PEPPER = (0, 140, 255)   # Amber/Orange (BGR)
    COLOR_RVIN        = (211, 0, 148)   # Purple/Violet (BGR)

    def __init__(self, original_image: np.ndarray):
        if len(original_image.shape) == 2:
            self.base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.base = original_image.copy()

    def create_color_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = self.base.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[mask == 1] = self.COLOR_SALT_PEPPER
        color_mask[mask == 2] = self.COLOR_RVIN
        return color_mask

    def create_overlay(self, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        color_mask = self.create_color_mask(mask)
        has_noise = (mask > 0).astype(np.uint8)
        overlay = self.base.copy()
        blended = cv2.addWeighted(self.base, 1 - alpha, color_mask, alpha, 0)
        overlay[has_noise == 1] = blended[has_noise == 1]
        return overlay

    def generate_full_visualization(
        self, mask: np.ndarray, severity_report: dict, output_path: str
    ) -> dict:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        stem = out_p.stem
        mask_path = out_p.parent / f"{stem}_mask.png"
        overlay_path = out_p.parent / f"{stem}_overlay.png"

        color_mask = self.create_color_mask(mask)
        cv2.imwrite(str(mask_path), color_mask)

        overlay = self.create_overlay(mask)
        cv2.imwrite(str(overlay_path), overlay)

        # Annotated image with contours
        annotated = overlay.copy()
        for cls_id, color, label in [
            (1, self.COLOR_SALT_PEPPER, "Salt & Pepper"),
            (2, self.COLOR_RVIN, "RVIN"),
        ]:
            binary = (mask == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 60:
                    cv2.drawContours(annotated, [cnt], -1, color, 1)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)

        cv2.imwrite(str(out_p), annotated)

        return {
            "annotated_path": str(out_p),
            "overlay_path": str(overlay_path),
            "mask_path": str(mask_path),
        }
