import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from backend.model_loader import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that the trained CT noise model can be loaded."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "model" / "best_model.pth",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for loading, for example cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(str(args.model), device=args.device)

    print()
    print("Model Loaded Successfully!")
    print(f"Checkpoint: {args.model}")
    print(f"Device: {args.device}")
    print(f"Model class: {model.__class__.__name__}")


if __name__ == "__main__":
    main()
