def calculate_severity_model4(probabilities: dict) -> dict:
    """
    Severity and diagnostic assessment for Model 4 (NoiseCNN).
    Detects Quantization Noise and Periodic Noise.
    """
    clean_prob = probabilities.get("clean", 0.0)
    quant_prob = probabilities.get("quantization", 0.0)
    periodic_prob = probabilities.get("periodic", 0.0)

    total_noise_prob = round((quant_prob + periodic_prob) * 100, 2)
    clean_pct = round(clean_prob * 100, 2)
    quant_pct = round(quant_prob * 100, 2)
    periodic_pct = round(periodic_prob * 100, 2)

    # Dominant class
    if clean_prob >= quant_prob and clean_prob >= periodic_prob:
        predicted_class = "Clean"
        confidence = clean_pct
    elif quant_prob >= periodic_prob:
        predicted_class = "Quantization Noise"
        confidence = quant_pct
    else:
        predicted_class = "Periodic Noise"
        confidence = periodic_pct

    def get_level(pct: float) -> str:
        if pct < 10.0:
            return "NONE"
        elif pct < 35.0:
            return "MILD"
        elif pct < 65.0:
            return "MODERATE"
        elif pct < 85.0:
            return "SEVERE"
        else:
            return "CRITICAL"

    return {
        "model": "Model 4",
        "architecture": "NoiseCNN",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "noise": {
            "quantization": {
                "probability": quant_prob,
                "severity_percentage": quant_pct,
                "severity_level": get_level(quant_pct),
            },
            "periodic": {
                "probability": periodic_prob,
                "severity_percentage": periodic_pct,
                "severity_level": get_level(periodic_pct),
            },
            "clean": {
                "probability": clean_prob,
                "severity_percentage": clean_pct,
                "severity_level": "OPTIMAL" if clean_pct >= 70 else "REDUCED",
            }
        },
        "summary": {
            "total_noise_percentage": total_noise_prob,
            "total_noise_level": get_level(total_noise_prob),
            "dominant_noise": predicted_class if predicted_class != "Clean" else "None",
        },
    }
