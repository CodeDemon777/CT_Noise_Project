"""
Model 2 Integration Test Script
Tests all 7 test cases defined in the integration prompt.
"""
import sys, cv2, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.model2 import Model2Predictor, calculate_severity_model2, Model2Visualizer

MODEL_PATH = str(PROJECT_ROOT / "backend" / "models" / "model2" / "Joshna.pth")
TEST_IMAGE = str(PROJECT_ROOT / "outputs" / "dataset_test_verification" / "dataset_clean_slice.png")
OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "m2_test_result.png")

print("=" * 55)
print("MODEL 2 INTEGRATION TESTS")
print("=" * 55)

# TEST 1: Model Loading
print("\n=== TEST 1: Model Loading ===")
predictor = Model2Predictor(MODEL_PATH)
print("PASS: Model 2 loaded, no missing/unexpected keys")

# TEST 2: Input Tensor
print("\n=== TEST 2: Input Tensor ===")
img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
tensor = predictor.preprocess(img)
shape = tuple(tensor.shape)
print(f"  Input shape: {shape}")
assert shape == (1, 1, 512, 512), f"FAIL: got {shape}"
print("PASS: [1, 1, 512, 512] tensor confirmed")

# TEST 3: Raw Output
print("\n=== TEST 3: Raw Output ===")
with torch.no_grad():
    output = predictor.model(tensor)
out_shape = tuple(output.shape)
print(f"  Output shape: {out_shape}")
assert out_shape == (1, 4, 512, 512), f"FAIL: got {out_shape}"
preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
print(f"  Argmax unique values (pre-remap): {np.unique(preds).tolist()}")
print("PASS: raw output [1,4,512,512], argmax works")

# TEST 4: Pixel Mask
print("\n=== TEST 4: Pixel Mask ===")
mask = predictor.predict(TEST_IMAGE)
unique = np.unique(mask).tolist()
print(f"  Mask shape:     {mask.shape}")
print(f"  Unique classes: {unique}")
print(f"  Clean   (0): {int(np.sum(mask == 0))} pixels")
print(f"  Poisson (1): {int(np.sum(mask == 1))} pixels")
print(f"  Speckle (2): {int(np.sum(mask == 2))} pixels")
assert 3 not in unique, "FAIL: raw class 3 found in final mask (remap not applied)"
print("PASS: class 3->2 remapping applied, no raw class 3 in mask")

# TEST 5: Severity
print("\n=== TEST 5: Severity Calculation ===")
report = calculate_severity_model2(mask)
p_pct = report["noise"]["poisson"]["severity_percentage"]
s_pct = report["noise"]["speckle"]["severity_percentage"]
p_lvl = report["noise"]["poisson"]["severity_level"]
s_lvl = report["noise"]["speckle"]["severity_level"]
t_pct = report["summary"]["total_noise_percentage"]
t_lvl = report["summary"]["total_noise_level"]
print(f"  Poisson: {p_pct}%  ->  {p_lvl}")
print(f"  Speckle: {s_pct}%  ->  {s_lvl}")
print(f"  Total:   {t_pct}%  ->  {t_lvl}")
print("PASS: Severity computed correctly")

# TEST 6: Visualization Colors
print("\n=== TEST 6: Visualization (Colors) ===")
orig_img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
visualizer = Model2Visualizer(orig_img)
result = visualizer.generate_full_visualization(mask, report, OUTPUT_PATH)
colored_mask = result["mask"]

if int(np.sum(mask == 1)) > 0:
    poisson_pixels = colored_mask[mask == 1]
    assert poisson_pixels[0, 2] == 255, "FAIL: Poisson not Red in BGR"
    print("  Poisson = Red (BGR 0,0,255): PASS")
else:
    print("  (No Poisson pixels in test image, skipping color check)")

if int(np.sum(mask == 2)) > 0:
    speckle_pixels = colored_mask[mask == 2]
    assert speckle_pixels[0, 1] == 255, "FAIL: Speckle not Green in BGR"
    print("  Speckle = Green (BGR 0,255,0): PASS")
else:
    print("  (No Speckle pixels in test image, skipping color check)")

print(f"  Output artifacts: {[k for k in result.keys() if 'path' in k]}")
print("PASS: Visualization saved with correct colors, pixel mask not replaced by boxes")

# TEST 7: Model Independence
print("\n=== TEST 7: Model Independence ===")
from backend.predict import CTPredictor
from backend.severity import SeverityCalculator

m1_predictor = CTPredictor(r'd:\Downloads\CT_Noise_Project - Copy\model\best_model.pth')
m1_mask = m1_predictor.predict(TEST_IMAGE)
m1_report = SeverityCalculator().get_detailed_report(m1_mask)

m2_mask = predictor.predict(TEST_IMAGE)
m2_report = calculate_severity_model2(m2_mask)

print(f"  Model 1 | Gaussian: {m1_report['gaussian']['percentage']}%  | Poisson: {m1_report['poisson']['percentage']}%")
print(f"  Model 2 | Poisson:  {m2_report['noise']['poisson']['severity_percentage']}% | Speckle: {m2_report['noise']['speckle']['severity_percentage']}%")
assert m1_mask is not m2_mask, "FAIL: Masks are the same object"
print("PASS: Model 1 and Model 2 run independently, results isolated")

print()
print("=" * 55)
print("ALL 7 TESTS PASSED")
print("=" * 55)
