"""
Severity Module
Calculates noise severity percentages from predicted mask
"""

import numpy as np
from typing import Dict, Tuple


class SeverityCalculator:
    """
    Calculates severity of different noise types
    """
    
    CLASS_GAUSSIAN = 1
    CLASS_POISSON = 2
    
    @staticmethod
    def calculate_severity(mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate severity percentages for each noise type
        
        Args:
            mask: Predicted segmentation mask (H, W) with class indices
        
        Returns:
            Dictionary with severity percentages:
            {
                "gaussian": float,  # 0-100
                "poisson": float,   # 0-100
                "total": float,     # Combined noise percentage
            }
        """
        total_pixels = mask.size
        
        # Count pixels for each class
        gaussian_pixels = np.sum(mask == SeverityCalculator.CLASS_GAUSSIAN)
        poisson_pixels = np.sum(mask == SeverityCalculator.CLASS_POISSON)
        
        # Calculate percentages
        gaussian_severity = (gaussian_pixels / total_pixels) * 100
        poisson_severity = (poisson_pixels / total_pixels) * 100
        total_severity = ((gaussian_pixels + poisson_pixels) / total_pixels) * 100
        
        result = {
            "gaussian": round(gaussian_severity, 2),
            "poisson": round(poisson_severity, 2),
            "total": round(total_severity, 2),
            "pixels": {
                "gaussian": int(gaussian_pixels),
                "poisson": int(poisson_pixels),
                "clean": int(np.sum(mask == 0)),
                "total": int(total_pixels),
            }
        }
        
        return result
    
    @staticmethod
    def classify_severity_level(percentage: float) -> str:
        """
        Classify severity as None, Mild, Moderate, Severe, or Critical
        
        Args:
            percentage: Severity percentage
        
        Returns:
            Severity level string
        """
        if percentage <= 0:
            return "None"
        elif percentage < 5:
            return "Mild"
        elif percentage < 15:
            return "Moderate"
        elif percentage < 30:
            return "Severe"
        else:
            return "Critical"
    
    @staticmethod
    def get_detailed_report(mask: np.ndarray) -> Dict:
        """
        Get detailed severity report with classifications
        
        Args:
            mask: Predicted segmentation mask
        
        Returns:
            Detailed report dictionary
        """
        severity = SeverityCalculator.calculate_severity(mask)
        
        report = {
            "gaussian": {
                "percentage": severity["gaussian"],
                "level": SeverityCalculator.classify_severity_level(severity["gaussian"]),
                "pixels": severity["pixels"]["gaussian"],
            },
            "poisson": {
                "percentage": severity["poisson"],
                "level": SeverityCalculator.classify_severity_level(severity["poisson"]),
                "pixels": severity["pixels"]["poisson"],
            },
            "summary": {
                "total_noise_percentage": severity["total"],
                "total_noise_level": SeverityCalculator.classify_severity_level(severity["total"]),
                "total_pixels": severity["pixels"]["total"],
                "clean_pixels": severity["pixels"]["clean"],
            },
            "pixels": severity["pixels"],
        }
        
        return report
    
    @staticmethod
    def print_report(report: Dict) -> None:
        """
        Print formatted severity report
        
        Args:
            report: Report dictionary from get_detailed_report()
        """
        print("\n" + "="*50)
        print("CT NOISE SEVERITY REPORT")
        print("="*50)
        
        print("\n📊 GAUSSIAN NOISE:")
        print(f"   Percentage: {report['gaussian']['percentage']}%")
        print(f"   Level: {report['gaussian']['level']}")
        print(f"   Pixels: {report['gaussian']['pixels']}")
        
        print("\n📊 POISSON NOISE:")
        print(f"   Percentage: {report['poisson']['percentage']}%")
        print(f"   Level: {report['poisson']['level']}")
        print(f"   Pixels: {report['poisson']['pixels']}")
        
        print("\n📈 SUMMARY:")
        print(f"   Total Noise: {report['summary']['total_noise_percentage']}%")
        print(f"   Level: {report['summary']['total_noise_level']}")
        print(f"   Clean Pixels: {report['summary']['clean_pixels']}")
        print(f"   Total Pixels: {report['summary']['total_pixels']}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test severity calculation
    test_mask = np.random.randint(0, 3, (512, 512))
    
    calculator = SeverityCalculator()
    report = calculator.get_detailed_report(test_mask)
    calculator.print_report(report)
    
    print("Report structure:")
    print(report)
