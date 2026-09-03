import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from backend.predict import CTPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CT noise inference tests on one image and optional controlled variants."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent / "test_ct_image.png",
        help="Path to the input CT image.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "backend" / "models" / "model1" / "best_model.pth",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference, for example cpu or cuda.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=18.0,
        help="Standard deviation used for the Gaussian-controlled test image.",
    )
    parser.add_argument(
        "--poisson-scale",
        type=float,
        default=1.0,
        help="Scale factor used before Poisson sampling.",
    )
    parser.add_argument(
        "--skip-controlled",
        action="store_true",
        help="Only run the clean image test and skip controlled noise variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder where test artifacts and summary JSON will be written.",
    )
    return parser.parse_args()


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_poisson_noise(image: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.clip(image.astype(np.float32) * scale, 0, 255)
    sampled = np.random.poisson(scaled).astype(np.float32)
    return np.clip(sampled / max(scale, 1e-6), 0, 255).astype(np.uint8)


def summarize_prediction(mask: np.ndarray) -> tuple[list[int], dict[str, int], dict[str, float]]:
    unique, counts = np.unique(mask, return_counts=True)
    total = int(mask.size)

    count_map = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
    pct_map = {int(cls): float(cnt) / total * 100.0 for cls, cnt in zip(unique, counts)}

    return (
        unique.tolist(),
        {
            "clean": count_map.get(0, 0),
            "gaussian": count_map.get(1, 0),
            "poisson": count_map.get(2, 0),
        },
        {
            "clean": round(pct_map.get(0, 0.0), 2),
            "gaussian": round(pct_map.get(1, 0.0), 2),
            "poisson": round(pct_map.get(2, 0.0), 2),
        },
    )


def save_prediction_preview(mask: np.ndarray, output_path: Path) -> None:
    preview = (mask.astype(np.uint8) * 100).astype(np.uint8)
    cv2.imwrite(str(output_path), preview)


def run_case(
    predictor: CTPredictor,
    case_name: str,
    image: np.ndarray,
    output_dir: Path,
) -> dict:
    input_path = output_dir / f"{case_name}.png"
    pred_path = output_dir / f"{case_name}_pred.png"

    cv2.imwrite(str(input_path), image)
    pred = predictor.predict(str(input_path))
    unique_classes, counts, percentages = summarize_prediction(pred)
    save_prediction_preview(pred, pred_path)

    print(f"=== {case_name.upper()} ===")
    print(f"Unique Classes: {unique_classes}")
    print(f"Clean: {percentages['clean']:.2f}%")
    print(f"Gaussian: {percentages['gaussian']:.2f}%")
    print(f"Poisson: {percentages['poisson']:.2f}%")
    print()

    return {
        "case": case_name,
        "image_path": str(input_path),
        "pred_path": str(pred_path),
        "unique_classes": unique_classes,
        "counts": counts,
        "percentages": percentages,
    }


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).parent / "outputs" / "reports" / f"inference_test_{timestamp}"


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    if not args.image.exists():
        fallback = Path(__file__).parent / "test_ct_image.png"
        if fallback.exists():
            print(f"⚠️ Specified image '{args.image}' not found. Falling back to default: '{fallback}'")
            args.image = fallback
        else:
            from scripts.generate_realistic_ct import create_realistic_ct_phantom
            print(f"⚠️ Image not found. Generating default phantom image at '{fallback}'...")
            create_realistic_ct_phantom(str(fallback))
            args.image = fallback
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== MODEL SETUP ===")
    predictor = CTPredictor(str(args.model), device=args.device)
    print(f"Checkpoint: {args.model}")
    print(f"Device: {args.device}")
    print()

    base_image = predictor.read_ct_image(str(args.image))
    results = [run_case(predictor, "clean_ct", base_image, output_dir)]

    if not args.skip_controlled:
        gaussian_image = add_gaussian_noise(base_image, sigma=args.gaussian_sigma)
        poisson_image = add_poisson_noise(base_image, scale=args.poisson_scale)
        combined_image = add_poisson_noise(gaussian_image, scale=args.poisson_scale)

        results.append(run_case(predictor, "gaussian_ct", gaussian_image, output_dir))
        results.append(run_case(predictor, "poisson_ct", poisson_image, output_dir))
        results.append(run_case(predictor, "gaussian_poisson_ct", combined_image, output_dir))

    summary = {
        "model_path": str(args.model),
        "input_image": str(args.image),
        "device": args.device,
        "gaussian_sigma": args.gaussian_sigma,
        "poisson_scale": args.poisson_scale,
        "results": results,
    }
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    print(f"Artifacts folder: {output_dir}")


if __name__ == "__main__":
    main()
