"""
Generate CT Test Images from Ground Truth Dataset with Model-Specific Noise Injections.

Supports extracting clean CT slices from HDF5 volumes and injecting specific noise types
for all 4 models:
  - Model 1 (UNet):          Gaussian Noise, Poisson Noise, and Localized Combined
  - Model 2 (Attention UNet): Poisson Noise, Speckle Noise, and Localized Combined
  - Model 3 (DeepLabV3+):     Salt & Pepper Noise, RVIN (Random-Valued Impulse), and Localized Combined
  - Model 4 (NoiseCNN):       Quantization Noise, Periodic Noise
"""

import argparse
import sys
from pathlib import Path
import cv2
import h5py
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def extract_clean_slice(hdf5_path: str, slice_idx: int = 64, target_size: int = 512) -> np.ndarray:
    """
    Extract a 2D slice from an HDF5 3D CT volume and normalize to [0, 255] uint8.
    """
    h5_file = Path(hdf5_path)
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 dataset file not found: {hdf5_path}")

    with h5py.File(str(h5_file), "r") as f:
        # Expected shape: (128, 362, 362) or (D, H, W)
        dataset = f["data"]
        total_slices = dataset.shape[0]
        actual_idx = min(max(0, slice_idx), total_slices - 1)
        raw_slice = dataset[actual_idx, :, :]

    # Normalize slice to [0, 255]
    min_val, max_val = float(raw_slice.min()), float(raw_slice.max())
    denom = max_val - min_val if (max_val - min_val) > 1e-8 else 1.0
    norm_slice = ((raw_slice - min_val) / denom) * 255.0
    img_uint8 = np.clip(norm_slice, 0, 255).astype(np.uint8)

    if (img_uint8.shape[0], img_uint8.shape[1]) != (target_size, target_size):
        img_uint8 = cv2.resize(img_uint8, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return img_uint8


# =====================================================================
# NOISE INJECTION FUNCTIONS FOR ALL 4 MODELS
# =====================================================================

def add_gaussian_noise(image: np.ndarray, std_dev: float = 30.0) -> np.ndarray:
    """Model 1 Noise: Additive zero-mean Gaussian electronic noise."""
    noise = np.random.normal(0.0, std_dev, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Model 1 & Model 2 Noise: Poisson photon shot noise."""
    scaled = np.clip(image.astype(np.float32) * scale, 0, 255)
    poisson = np.random.poisson(scaled).astype(np.float32)
    return np.clip(poisson / max(scale, 1e-6), 0, 255).astype(np.uint8)


def add_speckle_noise(image: np.ndarray, std_dev: float = 0.45) -> np.ndarray:
    """Model 2 Noise: Multiplicative Speckle noise."""
    noise = np.random.normal(0.0, std_dev, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) * (1.0 + noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.08) -> np.ndarray:
    """Model 3 Noise: Salt & Pepper impulse noise (bipolar min/max impulses)."""
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    coords_salt = [np.random.randint(0, dim, num_salt) for dim in image.shape]
    noisy[tuple(coords_salt)] = 255

    coords_pepper = [np.random.randint(0, dim, num_pepper) for dim in image.shape]
    noisy[tuple(coords_pepper)] = 0
    return noisy


def add_rvin_noise(image: np.ndarray, amount: float = 0.08) -> np.ndarray:
    """Model 3 Noise: Random-Valued Impulse Noise (RVIN: random uniform value impulse)."""
    noisy = image.copy()
    num_noisy = int(amount * image.size)
    coords = [np.random.randint(0, dim, num_noisy) for dim in image.shape]
    random_vals = np.random.randint(0, 256, num_noisy, dtype=np.uint8)
    noisy[tuple(coords)] = random_vals
    return noisy


def add_quantization_noise(image: np.ndarray, levels: int = 6) -> np.ndarray:
    """Model 4 Noise: Bit-depth reduction / step quantization noise."""
    step = 256.0 / max(levels, 2)
    quantized = np.floor(image.astype(np.float32) / step) * step + (step / 2.0)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def add_periodic_noise(image: np.ndarray, amplitude: float = 80.0, freq_x: float = 0.01, freq_y: float = 0.0) -> np.ndarray:
    """Model 4 Noise: Periodic sinusoidal / stripe interference artifact."""
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    pattern = amplitude * np.sin(2 * np.pi * (freq_x * x + freq_y * y))
    noisy = image.astype(np.float32) + pattern
    return np.clip(noisy, 0, 255).astype(np.uint8)


def create_localized_dual_noise(base_img: np.ndarray, noise1_fn, noise2_fn) -> np.ndarray:
    """
    Creates a CT image with two distinct localized circular noise zones (left and right lung/tissue areas)
    to verify multi-class spatial noise segmentation.
    """
    img = base_img.copy().astype(np.float32)
    h, w = img.shape[:2]

    # Left lobe mask
    mask1 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask1, (int(w * 0.35), int(h * 0.38)), int(w * 0.18), 255, -1)

    # Right lobe mask
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask2, (int(w * 0.65), int(h * 0.38)), int(w * 0.18), 255, -1)

    noisy1 = noise1_fn(base_img).astype(np.float32)
    noisy2 = noise2_fn(base_img).astype(np.float32)

    img = np.where(mask1 == 255, noisy1, img)
    img = np.where(mask2 == 255, noisy2, img)
    return np.clip(img, 0, 255).astype(np.uint8)


# =====================================================================
# MAIN GENERATION PIPELINE
# =====================================================================

def generate_dataset_test_suite(
    hdf5_path: str = "dataset/ground_truth_train/ground_truth_train_000.hdf5",
    slice_idx: int = 64,
    output_dir: str = "outputs/dataset_test_images"
) -> dict:
    """
    Extracts a clean slice and creates full test images with respective noises for all 4 models.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"📖 Loading clean slice {slice_idx} from: {hdf5_path}")
    clean_slice = extract_clean_slice(hdf5_path, slice_idx=slice_idx)
    
    clean_file = out_path / "dataset_clean.png"
    cv2.imwrite(str(clean_file), clean_slice)
    print(f"  [+] Saved Clean CT: {clean_file}")

    # Model 1 (UNet): Gaussian & Poisson
    m1_gauss = add_gaussian_noise(clean_slice, std_dev=30.0)
    m1_poisson = add_poisson_noise(clean_slice, scale=0.9)
    m1_dual = create_localized_dual_noise(
        clean_slice,
        lambda im: add_gaussian_noise(im, std_dev=32.0),
        lambda im: add_poisson_noise(im, scale=0.8)
    )
    cv2.imwrite(str(out_path / "m1_gaussian.png"), m1_gauss)
    cv2.imwrite(str(out_path / "m1_poisson.png"), m1_poisson)
    cv2.imwrite(str(out_path / "m1_dual_gaussian_poisson.png"), m1_dual)
    print("  [+] Model 1 test images generated (Gaussian, Poisson, Dual)")

    # Model 2 (Attention UNet): Poisson & Speckle
    m2_poisson = add_poisson_noise(clean_slice, scale=1.0)
    m2_speckle = add_speckle_noise(clean_slice, std_dev=0.45)
    m2_dual = create_localized_dual_noise(
        clean_slice,
        lambda im: add_poisson_noise(im, scale=1.0),
        lambda im: add_speckle_noise(im, std_dev=0.45)
    )
    cv2.imwrite(str(out_path / "m2_poisson.png"), m2_poisson)
    cv2.imwrite(str(out_path / "m2_speckle.png"), m2_speckle)
    cv2.imwrite(str(out_path / "m2_dual_poisson_speckle.png"), m2_dual)
    print("  [+] Model 2 test images generated (Poisson, Speckle, Dual)")

    # Model 3 (DeepLabV3+): Salt & Pepper & RVIN
    m3_sp = add_salt_and_pepper_noise(clean_slice, amount=0.08)
    m3_rvin = add_rvin_noise(clean_slice, amount=0.08)
    m3_dual = create_localized_dual_noise(
        clean_slice,
        lambda im: add_salt_and_pepper_noise(im, amount=0.08),
        lambda im: add_rvin_noise(im, amount=0.08)
    )
    cv2.imwrite(str(out_path / "m3_salt_pepper.png"), m3_sp)
    cv2.imwrite(str(out_path / "m3_rvin.png"), m3_rvin)
    cv2.imwrite(str(out_path / "m3_dual_sp_rvin.png"), m3_dual)
    print("  [+] Model 3 test images generated (Salt & Pepper, RVIN, Dual)")

    # Model 4 (NoiseCNN): Quantization & Periodic
    m4_quant = add_quantization_noise(clean_slice, levels=6)
    m4_periodic = add_periodic_noise(clean_slice, amplitude=80.0, freq_x=0.01, freq_y=0.0)
    cv2.imwrite(str(out_path / "m4_quantization.png"), m4_quant)
    cv2.imwrite(str(out_path / "m4_periodic.png"), m4_periodic)
    print("  [+] Model 4 test images generated (Quantization, Periodic)")

    print(f"\n✅ All dataset test images saved to: {out_path.resolve()}")
    return {
        "clean": str(clean_file),
        "output_dir": str(out_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset test images with noise for all 4 models.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="dataset/ground_truth_train/ground_truth_train_000.hdf5",
        help="Path to HDF5 ground truth dataset file."
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=64,
        help="Slice index to extract (0-127)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dataset_test_images",
        help="Directory to save generated test images."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset_test_suite(
        hdf5_path=args.hdf5,
        slice_idx=args.slice,
        output_dir=args.output_dir
    )
