import numpy as np


def calculate_severity_model3(mask: np.ndarray) -> dict:
    """
    Severity assessment for Model 3 (DeepLabV3+).
    Detects Salt & Pepper Noise and RVIN (Random-Valued Impulse Noise).
    Mask classes:
      0 = Clean / Background
      1 = Salt & Pepper
      2 = RVIN
    """
    total_pixels = mask.size
    clean_px = int(np.sum(mask == 0))
    sp_px = int(np.sum(mask == 1))
    rvin_px = int(np.sum(mask == 2))

    sp_pct = round((sp_px / total_pixels) * 100, 2)
    rvin_pct = round((rvin_px / total_pixels) * 100, 2)
    total_noise_pct = round(((sp_px + rvin_px) / total_pixels) * 100, 2)

    def get_level(pct: float) -> str:
        if pct == 0.0:
            return "NONE"
        elif pct < 5.0:
            return "MILD"
        elif pct < 15.0:
            return "MODERATE"
        elif pct < 30.0:
            return "SEVERE"
        else:
            return "CRITICAL"

    return {
        "model": "Model 3",
        "architecture": "DeepLabV3+",
        "pixels": {
            "total": total_pixels,
            "clean": clean_px,
            "salt_pepper": sp_px,
            "rvin": rvin_px,
        },
        "noise": {
            "salt_pepper": {
                "pixels": sp_px,
                "severity_percentage": sp_pct,
                "severity_level": get_level(sp_pct),
            },
            "rvin": {
                "pixels": rvin_px,
                "severity_percentage": rvin_pct,
                "severity_level": get_level(rvin_pct),
            },
        },
        "summary": {
            "total_noise_percentage": total_noise_pct,
            "total_noise_level": get_level(total_noise_pct),
            "dominant_noise": "Salt & Pepper" if sp_px >= rvin_px else "RVIN",
        },
    }
