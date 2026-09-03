"""
Model 2 Severity Calculator - Poisson + Speckle Noise
"""

import numpy as np
from typing import Optional, Dict


# Class mapping
# 0 = Clean
# 1 = Poisson
# 2 = Speckle

SEVERITY_THRESHOLDS = {
    "mild": 0.5,
    "moderate": 5.0,
}


def _get_severity_level(percentage: float) -> str:
    """Return severity level string based on percentage."""
    if percentage < SEVERITY_THRESHOLDS["mild"]:
        return "MILD"
    elif percentage < SEVERITY_THRESHOLDS["moderate"]:
        return "MODERATE"
    else:
        return "CRITICAL"


def calculate_severity_model2(
    pred_mask: np.ndarray,
    lung_mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate per-class severity for Model 2 predictions.

    Args:
        pred_mask: 2D integer array with values 0=Clean, 1=Poisson, 2=Speckle.
        lung_mask: Optional binary mask for lung region. When provided, percentages
                   are calculated relative to the lung region only.

    Returns:
        Severity report dictionary.
    """
    if lung_mask is not None:
        region = pred_mask[lung_mask > 0]
    else:
        region = pred_mask.flatten()

    total_pixels = region.size

    poisson_pixels = int(np.sum(region == 1))
    speckle_pixels = int(np.sum(region == 2))
    clean_pixels = int(np.sum(region == 0))

    poisson_pct = round(float(poisson_pixels) / total_pixels * 100.0, 2) if total_pixels > 0 else 0.0
    speckle_pct = round(float(speckle_pixels) / total_pixels * 100.0, 2) if total_pixels > 0 else 0.0
    total_noise_pct = round(poisson_pct + speckle_pct, 2)

    return {
        "model": "Model 2",
        "architecture": "Attention U-Net",
        "noise": {
            "poisson": {
                "class_id": 1,
                "pixel_count": poisson_pixels,
                "severity_percentage": poisson_pct,
                "severity_level": _get_severity_level(poisson_pct),
            },
            "speckle": {
                "class_id": 2,
                "pixel_count": speckle_pixels,
                "severity_percentage": speckle_pct,
                "severity_level": _get_severity_level(speckle_pct),
            },
        },
        "summary": {
            "clean_pixels": clean_pixels,
            "total_pixels": int(total_pixels),
            "total_noise_percentage": total_noise_pct,
            "total_noise_level": _get_severity_level(total_noise_pct),
        },
    }


def print_severity_report_model2(report: Dict) -> None:
    """Print a formatted severity report for Model 2."""
    print()
    print("=" * 50)
    print("MODEL 2 — ATTENTION U-NET NOISE SEVERITY REPORT")
    print("=" * 50)
    print()
    print("📊 POISSON NOISE:")
    print(f"   Percentage : {report['noise']['poisson']['severity_percentage']}%")
    print(f"   Level      : {report['noise']['poisson']['severity_level']}")
    print(f"   Pixels     : {report['noise']['poisson']['pixel_count']}")
    print()
    print("📊 SPECKLE NOISE:")
    print(f"   Percentage : {report['noise']['speckle']['severity_percentage']}%")
    print(f"   Level      : {report['noise']['speckle']['severity_level']}")
    print(f"   Pixels     : {report['noise']['speckle']['pixel_count']}")
    print()
    print("📈 SUMMARY:")
    print(f"   Total Noise : {report['summary']['total_noise_percentage']}%")
    print(f"   Level       : {report['summary']['total_noise_level']}")
    print(f"   Clean Pixels: {report['summary']['clean_pixels']}")
    print(f"   Total Pixels: {report['summary']['total_pixels']}")
    print("=" * 50)
    print()
