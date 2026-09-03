from pathlib import Path
import cv2
import numpy as np


class Model4Visualizer:
    """
    Visualization generator for Model 4 (NoiseCNN):
      Creates annotated CT view with clinical diagnostic HUD badge,
      FFT 2D spectrum (revealing periodic frequencies), and noise confidence map.
    """

    COLOR_QUANT = (0, 165, 255)   # Gold / Orange (BGR)
    COLOR_PERIODIC = (255, 0, 128) # Magenta / Electric Purple (BGR)
    COLOR_CLEAN = (0, 200, 100)    # Green (BGR)

    def __init__(self, original_image: np.ndarray):
        if len(original_image.shape) == 2:
            self.base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            self.gray = original_image.copy()
        else:
            self.base = original_image.copy()
            self.gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    def compute_fft_spectrum(self) -> np.ndarray:
        """Computes centered 2D Fourier Magnitude Spectrum for periodic noise visualization."""
        f = np.fft.fft2(self.gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        norm_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_spectrum = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_VIRIDIS)
        return color_spectrum

    def generate_full_visualization(
        self, probabilities: dict, severity_report: dict, output_path: str
    ) -> dict:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        stem = out_p.stem
        spectrum_path = out_p.parent / f"{stem}_spectrum.png"
        overlay_path = out_p.parent / f"{stem}_overlay.png"

        # 1. Save FFT Spectrum
        fft_vis = self.compute_fft_spectrum()
        cv2.imwrite(str(spectrum_path), fft_vis)

        # 2. Overlay with frequency blend
        overlay = cv2.addWeighted(self.base, 0.75, fft_vis, 0.25, 0)
        cv2.imwrite(str(overlay_path), overlay)

        # 3. Clinical Annotated Dashboard on Image
        annotated = self.base.copy()
        h, w = annotated.shape[:2]

        pred_class = severity_report.get("predicted_class", "Clean")
        confidence = severity_report.get("confidence", 0.0)
        q_pct = severity_report["noise"]["quantization"]["severity_percentage"]
        p_pct = severity_report["noise"]["periodic"]["severity_percentage"]

        # HUD Top Banner
        cv2.rectangle(annotated, (0, 0), (w, 65), (15, 23, 42), -1)
        cv2.line(annotated, (0, 65), (w, 65), (56, 189, 248), 2)

        accent = self.COLOR_CLEAN if pred_class == "Clean" else (self.COLOR_QUANT if "Quant" in pred_class else self.COLOR_PERIODIC)
        cv2.putText(annotated, f"MODEL 4: {pred_class.upper()}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, accent, 2)
        cv2.putText(annotated, f"Confidence: {confidence:.1f}% | Quant: {q_pct:.1f}% | Periodic: {p_pct:.1f}%", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 240), 1)

        cv2.imwrite(str(out_p), annotated)

        return {
            "annotated_path": str(out_p),
            "overlay_path": str(overlay_path),
            "spectrum_path": str(spectrum_path),
        }
