"""
Local Prediction Script - Phase 1
Tests the complete inference pipeline locally before deploying to backend/frontend
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from backend.predict import CTPredictor
from backend.severity import SeverityCalculator
from backend.visualization import CTVisualizer


def predict_and_visualize(image_path: str, model_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Complete inference pipeline: predict mask -> calculate severity -> visualize
    
    Args:
        image_path: Path to input CT image
        model_path: Path to model file
        output_dir: Output directory for results (default: ./outputs)
    
    Returns:
        Results dictionary with mask, severity, and output paths
    """
    output_dir = Path(output_dir or "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CT NOISE DETECTION - LOCAL INFERENCE")
    print("="*60)
    
    # Phase 1: Load image
    print("\n📷 STEP 1: Load Image")
    print("-" * 60)
    predictor = CTPredictor(model_path)
    original_image = predictor.read_ct_image(image_path)
    
    # Phase 2: Predict mask
    print("\n🔮 STEP 2: Predict Mask")
    print("-" * 60)
    predicted_mask = predictor.predict(image_path)
    print(f"✅ Prediction complete!")
    print(f"   Unique classes: {set(predicted_mask.flatten().tolist())}")
    
    # Phase 3: Calculate severity
    print("\n📊 STEP 3: Calculate Severity")
    print("-" * 60)
    calculator = SeverityCalculator()
    severity_report = calculator.get_detailed_report(predicted_mask)
    calculator.print_report(severity_report)
    
    # Phase 4: Visualize
    print("\n🎨 STEP 4: Generate Visualizations")
    print("-" * 60)
    visualizer = CTVisualizer(original_image)
    
    output_path = output_dir / "annotated" / "ct_result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visuals = visualizer.generate_full_visualization(
        predicted_mask,
        severity_report,
        str(output_path)
    )
    
    # Phase 5: Save mask
    print("\n💾 STEP 5: Save Outputs")
    print("-" * 60)
    import cv2
    mask_path = output_dir / "annotated" / "ct_mask.png"
    cv2.imwrite(str(mask_path), predicted_mask.astype('uint8') * 50)  # Scale for visibility
    print(f"💾 Saved mask to {mask_path}")
    
    overlay_path = output_path.parent / "ct_overlay.png"
    print(f"💾 Saved overlay to {overlay_path}")
    
    # Phase 6: Subplot Comparison (Matplotlib)
    print("\n📊 STEP 6: Matplotlib Subplot Comparison")
    print("-" * 60)
    plot_path = output_dir / "annotated" / "ct_comparison_plot.png"
    matplotlib_imported = False
    try:
        import matplotlib
        # Use Agg backend if headless (e.g. no display server) to save successfully
        import os
        if os.environ.get('DISPLAY', '') == '' and not os.name == 'nt':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib_imported = True
        
        # Select sample ID or file name
        idx_str = Path(image_path).name
        
        # Plotting (Matching user configuration)
        plt.figure(figsize=(20, 10))
        
        # Subplot 1: Diagnostic View
        plt.subplot(1, 2, 1)
        annotated_rgb = cv2.cvtColor(visuals["annotated"], cv2.COLOR_BGR2RGB)
        plt.imshow(annotated_rgb)
        plt.title(f"Diagnostic View: Annotated Noise Regions ({idx_str})")
        plt.axis('off')
        
        # Subplot 2: Segmentation Evidence Overlay
        plt.subplot(1, 2, 2)
        overlay_rgb = cv2.cvtColor(visuals["mask_overlay"], cv2.COLOR_BGR2RGB)
        plt.imshow(overlay_rgb)
        plt.title("Segmentation Evidence Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(str(plot_path), bbox_inches='tight', dpi=150)
        print(f"💾 Saved comparison plot to {plot_path}")
        
        # Attempt to show plot if interactive environment
        try:
            plt.show()
        except Exception as show_err:
            print(f"ℹ️ Could not open plot window (running in headless environment or no GUI): {show_err}")
            
    except ImportError:
        print("⚠️ matplotlib is not installed in the python environment. Skipping plot generation.")
        print("   To enable this feature, install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"❌ Failed to generate matplotlib plot: {e}")
    
    print("\n" + "="*60)
    print("✅ INFERENCE PIPELINE COMPLETE!")
    print("="*60)
    
    return {
        "original_image": original_image,
        "predicted_mask": predicted_mask,
        "severity_report": severity_report,
        "output_annotated": str(output_path),
        "output_mask": str(mask_path),
        "output_overlay": str(overlay_path),
        "output_comparison_plot": str(plot_path) if matplotlib_imported else None,
    }


def main():
    parser = argparse.ArgumentParser(description="CT Noise Detection - Local Inference")
    parser.add_argument("image", type=str, help="Path to CT image")
    parser.add_argument("--model", type=str, default=None, help="Path to model (default: model/best_model.pth)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve model path
    if args.model is None:
        args.model = Path(__file__).parent / "model" / "best_model.pth"
    
    # Run prediction
    results = predict_and_visualize(args.image, str(args.model), args.output)
    
    print("\n📁 Output Files:")
    print(f"   Annotated: {results['output_annotated']}")
    print(f"   Mask: {results['output_mask']}")
    print(f"   Overlay: {results['output_overlay']}")
    if results['output_comparison_plot']:
        print(f"   Comparison Plot: {results['output_comparison_plot']}")


if __name__ == "__main__":
    import numpy as np
    import cv2
    
    # If run without args, create test image
    if len(sys.argv) == 1:
        print("No arguments provided. Creating test image...")
        
        test_image_path = Path(__file__).parent / "test_ct_image.png"
        if not test_image_path.exists():
            test_image = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
            cv2.imwrite(str(test_image_path), test_image)
            print(f"✅ Created test image: {test_image_path}")
        
        model_path = Path(__file__).parent / "model" / "best_model.pth"
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("   Please move best_model.pth to model/ directory")
            sys.exit(1)
        
        results = predict_and_visualize(str(test_image_path), str(model_path))
    else:
        main()
