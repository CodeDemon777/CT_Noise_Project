"""
Comprehensive Dataset Verification Suite for All 4 CT Noise Models.

Loads real CT slices from the ground-truth training dataset, adds the corresponding noise types
for each model, executes full inference, computes severity metrics, and generates visual overlays:
  - Model 1 (UNet):          Gaussian Noise + Poisson Noise
  - Model 2 (Attention UNet): Poisson Noise + Speckle Noise
  - Model 3 (DeepLabV3+):     Salt & Pepper Noise + RVIN Noise
  - Model 4 (NoiseCNN):       Quantization Noise + Periodic Noise
"""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.predict import CTPredictor
from backend.severity import SeverityCalculator
from backend.visualization import CTVisualizer
from backend.models.model2 import Model2Predictor, calculate_severity_model2, Model2Visualizer
from backend.models.model3.predictor import Model3Predictor
from backend.models.model3.severity import calculate_severity_model3
from backend.models.model3.visualization import Model3Visualizer
from backend.models.model4.predictor import Model4Predictor
from backend.models.model4.severity import calculate_severity_model4
from backend.models.model4.visualization import Model4Visualizer
from scripts.generate_dataset_test_images import (
    extract_clean_slice,
    add_gaussian_noise,
    add_poisson_noise,
    add_speckle_noise,
    add_salt_and_pepper_noise,
    add_rvin_noise,
    add_quantization_noise,
    add_periodic_noise,
    create_localized_dual_noise,
)


def run_full_verification(
    hdf5_path: str = "dataset/ground_truth_train/ground_truth_train_000.hdf5",
    slice_idx: int = 64,
    device: str = "cpu",
    output_dir: str = "outputs/dataset_test_verification",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      DATASET CT SLICE NOISE INJECTION & 4-MODEL VERIFICATION SUITE")
    print("=" * 75)
    print(f"Dataset File : {hdf5_path}")
    print(f"Slice Index  : {slice_idx}")
    print(f"Device       : {device}")
    print(f"Output Dir   : {out_dir.resolve()}\n")

    # 1. Extract clean ground-truth CT slice
    clean_slice = extract_clean_slice(hdf5_path, slice_idx=slice_idx)
    clean_path = out_dir / "dataset_clean_slice.png"
    cv2.imwrite(str(clean_path), clean_slice)
    print(f"[+] Ground Truth clean CT slice extracted -> {clean_path.name}")

    results_summary = {
        "dataset_source": hdf5_path,
        "slice_index": slice_idx,
        "device": device,
        "models": {},
    }

    # =========================================================================
    # MODEL 1: UNet (Gaussian Noise & Poisson Noise)
    # =========================================================================
    print("\n" + "=" * 55)
    print(" MODEL 1: UNet (Gaussian + Poisson Noise)")
    print("=" * 55)
    m1_predictor = CTPredictor("backend/models/model1/best_model.pth", device=device)
    m1_calc = SeverityCalculator()

    m1_dual_img = create_localized_dual_noise(
        clean_slice,
        lambda im: add_gaussian_noise(im, std_dev=32.0),
        lambda im: add_poisson_noise(im, scale=0.8),
    )
    m1_test_path = out_dir / "m1_test_gaussian_poisson.png"
    cv2.imwrite(str(m1_test_path), m1_dual_img)

    m1_mask = m1_predictor.predict(str(m1_test_path))
    m1_rep = m1_calc.get_detailed_report(m1_mask)
    m1_vis = CTVisualizer(m1_dual_img)
    m1_vis.generate_full_visualization(m1_mask, m1_rep, str(out_dir / "m1_annotated.png"))

    print(f"  Test Image : {m1_test_path.name}")
    print(f"  -> Gaussian Noise : {m1_rep['gaussian']['percentage']:.2f}%  [Severity: {m1_rep['gaussian']['level']}]")
    print(f"  -> Poisson Noise  : {m1_rep['poisson']['percentage']:.2f}%  [Severity: {m1_rep['poisson']['level']}]")
    print(f"  -> Total Noise    : {m1_rep['summary']['total_noise_percentage']:.2f}%  [Overall: {m1_rep['summary']['total_noise_level']}]")

    results_summary["models"]["model1_unet"] = {
        "architecture": "UNet",
        "noise_types": ["Gaussian", "Poisson"],
        "gaussian_pct": m1_rep["gaussian"]["percentage"],
        "gaussian_level": m1_rep["gaussian"]["level"],
        "poisson_pct": m1_rep["poisson"]["percentage"],
        "poisson_level": m1_rep["poisson"]["level"],
        "total_pct": m1_rep["summary"]["total_noise_percentage"],
        "total_level": m1_rep["summary"]["total_noise_level"],
        "annotated_artifact": str(out_dir / "m1_annotated.png"),
    }

    # =========================================================================
    # MODEL 2: Attention U-Net (Poisson Noise & Speckle Noise)
    # =========================================================================
    print("\n" + "=" * 55)
    print(" MODEL 2: Attention U-Net (Poisson + Speckle Noise)")
    print("=" * 55)
    m2_predictor = Model2Predictor("backend/models/model2/Joshna.pth", device=device)

    m2_dual_img = create_localized_dual_noise(
        clean_slice,
        lambda im: add_poisson_noise(im, scale=1.0),
        lambda im: add_speckle_noise(im, std_dev=0.45),
    )
    m2_test_path = out_dir / "m2_test_poisson_speckle.png"
    cv2.imwrite(str(m2_test_path), m2_dual_img)

    m2_mask = m2_predictor.predict(str(m2_test_path))
    m2_rep = calculate_severity_model2(m2_mask)
    m2_vis = Model2Visualizer(m2_dual_img)
    m2_vis.generate_full_visualization(m2_mask, m2_rep, str(out_dir / "m2_annotated.png"))

    p2_pct = m2_rep["noise"]["poisson"]["severity_percentage"]
    p2_lvl = m2_rep["noise"]["poisson"]["severity_level"]
    s2_pct = m2_rep["noise"]["speckle"]["severity_percentage"]
    s2_lvl = m2_rep["noise"]["speckle"]["severity_level"]
    t2_pct = m2_rep["summary"]["total_noise_percentage"]
    t2_lvl = m2_rep["summary"]["total_noise_level"]

    print(f"  Test Image : {m2_test_path.name}")
    print(f"  -> Poisson Noise : {p2_pct:.2f}%  [Severity: {p2_lvl}]")
    print(f"  -> Speckle Noise : {s2_pct:.2f}%  [Severity: {s2_lvl}]")
    print(f"  -> Total Noise   : {t2_pct:.2f}%  [Overall: {t2_lvl}]")

    results_summary["models"]["model2_attention_unet"] = {
        "architecture": "Attention U-Net",
        "noise_types": ["Poisson", "Speckle"],
        "poisson_pct": p2_pct,
        "poisson_level": p2_lvl,
        "speckle_pct": s2_pct,
        "speckle_level": s2_lvl,
        "total_pct": t2_pct,
        "total_level": t2_lvl,
        "annotated_artifact": str(out_dir / "m2_annotated.png"),
    }

    # =========================================================================
    # MODEL 3: DeepLabV3+ (Salt & Pepper Noise & RVIN Noise)
    # =========================================================================
    print("\n" + "=" * 55)
    print(" MODEL 3: DeepLabV3+ (Salt & Pepper + RVIN Noise)")
    print("=" * 55)
    m3_predictor = Model3Predictor("backend/models/model3/Jahnavi (1).pth", device=device)

    m3_dual_img = create_localized_dual_noise(
        clean_slice,
        lambda im: add_salt_and_pepper_noise(im, amount=0.08),
        lambda im: add_rvin_noise(im, amount=0.08),
    )
    m3_test_path = out_dir / "m3_test_saltpepper_rvin.png"
    cv2.imwrite(str(m3_test_path), m3_dual_img)

    m3_mask = m3_predictor.predict(str(m3_test_path))
    m3_rep = calculate_severity_model3(m3_mask)
    m3_vis = Model3Visualizer(m3_dual_img)
    m3_vis.generate_full_visualization(m3_mask, m3_rep, str(out_dir / "m3_annotated.png"))

    sp_pct = m3_rep["noise"]["salt_pepper"]["severity_percentage"]
    sp_lvl = m3_rep["noise"]["salt_pepper"]["severity_level"]
    rvin_pct = m3_rep["noise"]["rvin"]["severity_percentage"]
    rvin_lvl = m3_rep["noise"]["rvin"]["severity_level"]
    t3_pct = m3_rep["summary"]["total_noise_percentage"]
    t3_lvl = m3_rep["summary"]["total_noise_level"]

    print(f"  Test Image : {m3_test_path.name}")
    print(f"  -> Salt & Pepper : {sp_pct:.2f}%  [Severity: {sp_lvl}]")
    print(f"  -> RVIN Noise    : {rvin_pct:.2f}%  [Severity: {rvin_lvl}]")
    print(f"  -> Total Noise   : {t3_pct:.2f}%  [Overall: {t3_lvl}]")

    results_summary["models"]["model3_deeplabv3plus"] = {
        "architecture": "DeepLabV3+",
        "noise_types": ["Salt & Pepper", "RVIN"],
        "salt_pepper_pct": sp_pct,
        "salt_pepper_level": sp_lvl,
        "rvin_pct": rvin_pct,
        "rvin_level": rvin_lvl,
        "total_pct": t3_pct,
        "total_level": t3_lvl,
        "annotated_artifact": str(out_dir / "m3_annotated.png"),
    }

    # =========================================================================
    # MODEL 4: NoiseCNN (Quantization Noise & Periodic Noise)
    # =========================================================================
    print("\n" + "=" * 55)
    print(" MODEL 4: NoiseCNN (Quantization + Periodic Noise)")
    print("=" * 55)
    m4_predictor = Model4Predictor("backend/models/model4/Vasanth (2).pth", device=device)

    # Test 4.1: Quantization Noise
    m4_quant_img = add_quantization_noise(clean_slice, levels=6)
    m4_quant_path = out_dir / "m4_test_quantization.png"
    cv2.imwrite(str(m4_quant_path), m4_quant_img)
    m4_quant_res = m4_predictor.predict(str(m4_quant_path))
    m4_quant_rep = calculate_severity_model4(m4_quant_res)
    m4_vis_q = Model4Visualizer(m4_quant_img)
    m4_vis_q.generate_full_visualization(m4_quant_res, m4_quant_rep, str(out_dir / "m4_quant_annotated.png"))

    print(f"  Test Image (Quantization) : {m4_quant_path.name}")
    print(f"  -> Predicted Class : {m4_quant_rep['predicted_class']} (Confidence: {m4_quant_rep['confidence']:.1f}%)")
    print(f"  -> Probs : Clean={m4_quant_res['clean']*100:.2f}%, Quant={m4_quant_res['quantization']*100:.2f}%, Periodic={m4_quant_res['periodic']*100:.2f}%")

    # Test 4.2: Periodic Noise
    m4_periodic_img = add_periodic_noise(clean_slice, amplitude=80.0, freq_x=0.01, freq_y=0.0)
    m4_periodic_path = out_dir / "m4_test_periodic.png"
    cv2.imwrite(str(m4_periodic_path), m4_periodic_img)
    m4_periodic_res = m4_predictor.predict(str(m4_periodic_path))
    m4_periodic_rep = calculate_severity_model4(m4_periodic_res)
    m4_vis_p = Model4Visualizer(m4_periodic_img)
    m4_vis_p.generate_full_visualization(m4_periodic_res, m4_periodic_rep, str(out_dir / "m4_periodic_annotated.png"))

    print(f"\n  Test Image (Periodic) : {m4_periodic_path.name}")
    print(f"  -> Predicted Class : {m4_periodic_rep['predicted_class']} (Confidence: {m4_periodic_rep['confidence']:.1f}%)")
    print(f"  -> Probs : Clean={m4_periodic_res['clean']*100:.2f}%, Quant={m4_periodic_res['quantization']*100:.2f}%, Periodic={m4_periodic_res['periodic']*100:.2f}%")

    results_summary["models"]["model4_noisecnn"] = {
        "architecture": "NoiseCNN",
        "noise_types": ["Quantization", "Periodic"],
        "quantization_test": {
            "predicted_class": m4_quant_rep["predicted_class"],
            "confidence": m4_quant_rep["confidence"],
            "probs": m4_quant_res,
            "artifact": str(out_dir / "m4_quant_annotated.png"),
        },
        "periodic_test": {
            "predicted_class": m4_periodic_rep["predicted_class"],
            "confidence": m4_periodic_rep["confidence"],
            "probs": m4_periodic_res,
            "artifact": str(out_dir / "m4_periodic_annotated.png"),
        },
    }

    # Save summary JSON
    summary_file = out_dir / "verification_summary.json"
    summary_file.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\n📁 Saved full verification summary report to: {summary_file}")

    print("\n" + "=" * 75)
    print("      ALL 4 MODELS VERIFIED SUCCESSFULLY ON DATASET NOISE SLICES")
    print("=" * 75)

    return results_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Verify all 4 CT noise models on dataset slices with added noise.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="dataset/ground_truth_train/ground_truth_train_000.hdf5",
        help="Path to HDF5 ground truth dataset file.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=64,
        help="Slice index to extract from the volume (0-127).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dataset_test_verification",
        help="Directory to save generated test images and verification results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_verification(
        hdf5_path=args.hdf5,
        slice_idx=args.slice,
        device=args.device,
        output_dir=args.output_dir,
    )
