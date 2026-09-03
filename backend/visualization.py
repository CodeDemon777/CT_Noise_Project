"""
Visualization Module
Creates annotated images with bounding boxes for detected noise regions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from scipy import ndimage


class CTVisualizer:
    """
    Creates visualizations of CT noise detection results
    """
    
    # Colors for different noise types
    GAUSSIAN_COLOR = (0, 0, 255)    # Red (BGR)
    POISSON_COLOR = (255, 0, 0)     # Blue (BGR)
    
    CLASS_GAUSSIAN = 1
    CLASS_POISSON = 2
    
    def __init__(self, original_image: np.ndarray):
        """
        Initialize visualizer with original image
        
        Args:
            original_image: Original CT image (grayscale)
        """
        self.original_image = original_image
        # Convert to RGB for visualization
        self.image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) \
            if len(original_image.shape) == 2 else original_image.copy()
    
    def get_bounding_boxes(self, mask: np.ndarray, class_id: int, min_size: int = 10) -> list:
        """
        Extract bounding boxes for a specific class in O(1) OpenCV contours without ndimage bottleneck.
        """
        class_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                bboxes.append((x, y, w, h))
        return bboxes
    
    def draw_boxes(self, mask: np.ndarray, line_thickness: int = 2, min_area: float = 50.0) -> Tuple[np.ndarray, list]:
        """
        Draw bounding boxes and region severity labels on image using contours.
        
        Args:
            mask: Segmentation mask
            line_thickness: Thickness of bounding box lines
            min_area: Minimum area of contour to filter out noise
        
        Returns:
            Tuple of (Annotated image, list of regions)
        """
        annotated = self.image_rgb.copy()
        total_pixels = mask.size
        regions = []
        box_id = 1
        
        # Class 1: Gaussian (Red BGR: GAUSSIAN_COLOR), Class 2: Poisson (Blue BGR: POISSON_COLOR)
        for class_idx in [self.CLASS_GAUSSIAN, self.CLASS_POISSON]:
            class_mask = (mask == class_idx).astype(np.uint8)
            name = "Gaussian" if class_idx == self.CLASS_GAUSSIAN else "Poisson"
            color = self.GAUSSIAN_COLOR if class_idx == self.CLASS_GAUSSIAN else self.POISSON_COLOR
            
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            drew_count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                region_pixels = np.sum(class_mask[y:y+h, x:x+w])
                severity = (region_pixels / total_pixels) * 100
                
                # Draw Box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, line_thickness)
                
                # Draw Label as sequence number (ensure label text is within the image bounds)
                label = f"{box_id}"
                text_y = max(y - 10, 15)
                cv2.putText(annotated, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                regions.append({
                    "id": box_id,
                    "type": name,
                    "percentage": round(severity, 2)
                })
                
                box_id += 1
                drew_count += 1
            
            print(f"🎨 Drew {drew_count} {name} contour boxes (min_area: {min_area})")
            
        return annotated, regions
    
    def add_severity_labels(self, annotated: np.ndarray, severity_report: Dict) -> np.ndarray:
        """
        Add severity text labels to image (no-op as totals are displayed under the image in the UI)
        """
        return annotated
    
    def create_mask_overlay(self, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Create semi-transparent mask overlay, keeping clean pixels clean and bright.
        
        Args:
            mask: Segmentation mask
            alpha: Transparency level (0-1)
        
        Returns:
            Image with mask overlay
        """
        result = self.image_rgb.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(self.image_rgb)
        colored_mask[mask == self.CLASS_GAUSSIAN] = self.GAUSSIAN_COLOR
        colored_mask[mask == self.CLASS_POISSON] = self.POISSON_COLOR
        
        # Blend only where mask is not 0 (i.e. noise regions)
        noise_mask = (mask > 0)
        if np.any(noise_mask):
            blended = cv2.addWeighted(self.image_rgb, 1 - alpha, colored_mask, alpha, 0)
            result[noise_mask] = blended[noise_mask]
            
        return result
    
    def create_side_by_side(self, annotated: np.ndarray, mask_overlay: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of original and annotated
        
        Args:
            annotated: Annotated image with boxes
            mask_overlay: Mask overlay image
        
        Returns:
            Side-by-side comparison image
        """
        # Ensure same height
        h = max(annotated.shape[0], mask_overlay.shape[0])
        
        # Resize if needed
        if annotated.shape[0] != h:
            annotated = cv2.resize(annotated, (annotated.shape[1], h))
        if mask_overlay.shape[0] != h:
            mask_overlay = cv2.resize(mask_overlay, (mask_overlay.shape[1], h))
        
        # Concatenate
        result = np.hstack([self.original_image[:, :, np.newaxis].repeat(3, axis=2), 
                           annotated, mask_overlay])
        
        return result
    
    def save_visualization(self, annotated: np.ndarray, output_path: str) -> None:
        """
        Save annotated image to file
        
        Args:
            annotated: Annotated image
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), annotated)
        print(f"💾 Saved visualization to {output_path}")
    
    def generate_full_visualization(self, mask: np.ndarray, severity_report: Dict, 
                                   output_path: str) -> Dict[str, Any]:
        """
        Generate complete visualization with all elements
        
        Args:
            mask: Segmentation mask
            severity_report: Severity report
            output_path: Output file path
        
        Returns:
            Dictionary with all generated visualizations
        """
        # Draw boxes
        annotated, regions = self.draw_boxes(mask)
        
        # Add severity labels
        annotated_labeled = self.add_severity_labels(annotated, severity_report)
        
        # Create mask overlay
        mask_overlay = self.create_mask_overlay(mask)
        
        # Save main result
        self.save_visualization(annotated_labeled, output_path)
        
        # Save overlay result (same directory, suffix _overlay.png)
        output_path_obj = Path(output_path)
        overlay_path = output_path_obj.parent / f"{output_path_obj.name.replace('_result.png', '')}_overlay.png"
        self.save_visualization(mask_overlay, str(overlay_path))
        
        result = {
            "annotated": annotated_labeled,
            "mask_overlay": mask_overlay,
            "original": self.original_image,
            "regions": regions,
        }
        
        return result


if __name__ == "__main__":
    # Test visualization
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    test_mask = np.random.randint(0, 3, (512, 512), dtype=np.uint8)
    
    visualizer = CTVisualizer(test_image)
    
    # Create dummy severity report
    severity_report = {
        "gaussian": {"percentage": 2.83, "level": "Mild"},
        "poisson": {"percentage": 1.79, "level": "Mild"},
    }
    
    output_path = Path(__file__).parent.parent / "outputs" / "annotated" / "test_visualization.png"
    visuals = visualizer.generate_full_visualization(test_mask, severity_report, str(output_path))
    print(f"Generated {len(visuals)} visualizations")
