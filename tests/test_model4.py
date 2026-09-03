"""
Integration & Unit Test Suite for Model 4 (NoiseCNN — Quantization + Periodic Noise).
"""
import unittest
import numpy as np
import torch
from pathlib import Path
import cv2

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.model4.model import NoiseCNN
from backend.models.model4.model_loader import load_model4
from backend.models.model4.predictor import Model4Predictor
from backend.models.model4.severity import calculate_severity_model4
from backend.models.model4.visualization import Model4Visualizer

MODEL4_PATH = PROJECT_ROOT / "backend" / "models" / "model4" / "Vasanth (2).pth"
TEST_IMG_PATH = PROJECT_ROOT / "outputs" / "dataset_test_verification" / "dataset_clean_slice.png"


class TestModel4(unittest.TestCase):

    def setUp(self):
        self.assertTrue(MODEL4_PATH.exists(), f"Model 4 file missing: {MODEL4_PATH}")

    def test_01_model_instantiation_and_forward(self):
        model = NoiseCNN(num_classes=3)
        dummy = torch.randn(1, 1, 128, 128)
        out = model(dummy)
        self.assertEqual(out.shape, (1, 3))

    def test_02_model_loader(self):
        model = load_model4(str(MODEL4_PATH), device="cpu")
        self.assertIsInstance(model, NoiseCNN)
        self.assertFalse(model.training)

    def test_03_predictor_inference(self):
        predictor = Model4Predictor(str(MODEL4_PATH), device="cpu")
        if not TEST_IMG_PATH.exists():
            img = (np.random.rand(512, 512) * 255).astype(np.uint8)
            cv2.imwrite(str(TEST_IMG_PATH), img)

        probs = predictor.predict(str(TEST_IMG_PATH))
        self.assertIn("clean", probs)
        self.assertIn("quantization", probs)
        self.assertIn("periodic", probs)
        total = probs["clean"] + probs["quantization"] + probs["periodic"]
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_04_severity_calculation(self):
        mock_probs = {"clean": 0.05, "quantization": 0.85, "periodic": 0.10}
        rep = calculate_severity_model4(mock_probs)

        self.assertEqual(rep["model"], "Model 4")
        self.assertEqual(rep["architecture"], "NoiseCNN")
        self.assertEqual(rep["predicted_class"], "Quantization Noise")
        self.assertEqual(rep["confidence"], 85.0)
        self.assertEqual(rep["noise"]["quantization"]["severity_level"], "CRITICAL")
        self.assertAlmostEqual(rep["summary"]["total_noise_percentage"], 95.0, places=1)

    def test_05_visualizer_generation(self):
        dummy_img = np.full((256, 256), 128, dtype=np.uint8)
        vis = Model4Visualizer(dummy_img)
        mock_probs = {"clean": 0.1, "quantization": 0.7, "periodic": 0.2}
        rep = calculate_severity_model4(mock_probs)

        out_path = Path("outputs/test_m4_out.png")
        paths = vis.generate_full_visualization(mock_probs, rep, str(out_path))

        self.assertTrue(Path(paths["annotated_path"]).exists())
        self.assertTrue(Path(paths["overlay_path"]).exists())
        self.assertTrue(Path(paths["spectrum_path"]).exists())


if __name__ == "__main__":
    unittest.main()
