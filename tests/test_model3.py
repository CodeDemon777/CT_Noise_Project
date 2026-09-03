"""
Integration & Unit Test Suite for Model 3 (DeepLabV3+ — Salt & Pepper + RVIN).
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

from backend.models.model3.model import DeepLabV3Plus
from backend.models.model3.model_loader import load_model3
from backend.models.model3.predictor import Model3Predictor
from backend.models.model3.severity import calculate_severity_model3
from backend.models.model3.visualization import Model3Visualizer

MODEL3_PATH = PROJECT_ROOT / "backend" / "models" / "model3" / "Jahnavi (1).pth"
TEST_IMG_PATH = PROJECT_ROOT / "outputs" / "dataset_test_verification" / "dataset_clean_slice.png"


class TestModel3(unittest.TestCase):

    def setUp(self):
        self.assertTrue(MODEL3_PATH.exists(), f"Model 3 file missing: {MODEL3_PATH}")

    def test_01_model_instantiation_and_forward(self):
        model = DeepLabV3Plus(num_classes=3)
        dummy = torch.randn(1, 1, 512, 512)
        out = model(dummy)
        self.assertEqual(out.shape, (1, 3, 512, 512))

    def test_02_model_loader(self):
        model = load_model3(str(MODEL3_PATH), device="cpu")
        self.assertIsInstance(model, DeepLabV3Plus)
        self.assertFalse(model.training)

    def test_03_predictor_inference(self):
        predictor = Model3Predictor(str(MODEL3_PATH), device="cpu")
        if not TEST_IMG_PATH.exists():
            # Generate a test grayscale slice
            img = (np.random.rand(512, 512) * 255).astype(np.uint8)
            cv2.imwrite(str(TEST_IMG_PATH), img)

        mask = predictor.predict(str(TEST_IMG_PATH))
        self.assertEqual(mask.shape, (512, 512))
        self.assertEqual(mask.dtype, np.uint8)
        # Unique classes must be subset of {0, 1, 2}
        classes = set(np.unique(mask))
        self.assertTrue(classes.issubset({0, 1, 2}))

    def test_04_severity_calculation(self):
        # Create a mock mask with 10% SP and 5% RVIN
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:10, :] = 1  # 10%
        mask[10:15, :] = 2  # 5%
        rep = calculate_severity_model3(mask)

        self.assertEqual(rep["model"], "Model 3")
        self.assertEqual(rep["architecture"], "DeepLabV3+")
        self.assertAlmostEqual(rep["noise"]["salt_pepper"]["severity_percentage"], 10.0, places=1)
        self.assertAlmostEqual(rep["noise"]["rvin"]["severity_percentage"], 5.0, places=1)
        self.assertAlmostEqual(rep["summary"]["total_noise_percentage"], 15.0, places=1)
        self.assertEqual(rep["noise"]["salt_pepper"]["severity_level"], "MODERATE")

    def test_05_visualizer_generation(self):
        dummy_img = np.full((512, 512), 128, dtype=np.uint8)
        vis = Model3Visualizer(dummy_img)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:150, 100:150] = 1
        mask[200:250, 200:250] = 2
        rep = calculate_severity_model3(mask)

        out_path = Path("outputs/test_m3_out.png")
        paths = vis.generate_full_visualization(mask, rep, str(out_path))

        self.assertTrue(Path(paths["annotated_path"]).exists())
        self.assertTrue(Path(paths["overlay_path"]).exists())
        self.assertTrue(Path(paths["mask_path"]).exists())


if __name__ == "__main__":
    unittest.main()
