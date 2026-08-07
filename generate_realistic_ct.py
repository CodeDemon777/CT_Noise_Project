import cv2
import numpy as np
from pathlib import Path

def create_realistic_ct_phantom(output_path: str):
    # 1. Start with a black background (512x512)
    img = np.zeros((512, 512), dtype=np.float32)
    
    # 2. Draw patient body outline (large ellipse in center)
    cv2.ellipse(img, (256, 256), (180, 140), 0, 0, 360, 100, -1)
    
    # 3. Draw lung cavities (two dark ellipses)
    cv2.ellipse(img, (180, 240), (60, 45), -15, 0, 360, 25, -1)
    cv2.ellipse(img, (332, 240), (60, 45), 15, 0, 360, 25, -1)
    
    # 4. Draw spine/bones (bright circle near top center and ribs)
    cv2.circle(img, (256, 350), 22, 190, -1) # spine
    cv2.circle(img, (256, 350), 12, 60, -1)  # spinal canal
    
    # Draw ribs around the ellipse border
    for angle in range(0, 360, 30):
        rad = np.deg2rad(angle)
        rx = int(256 + 175 * np.cos(rad))
        ry = int(256 + 135 * np.sin(rad))
        cv2.circle(img, (rx, ry), 6, 210, -1)

    # 5. Localize Gaussian Noise Region (right soft tissue area)
    mask_gaussian = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask_gaussian, (170, 150), 45, 255, -1)
    
    # Generate Gaussian Noise
    mean = 0
    std_dev = 25
    gaussian_noise = np.random.normal(mean, std_dev, (512, 512))
    
    # Apply Gaussian noise to that region
    img = np.where(mask_gaussian == 255, np.clip(img + gaussian_noise, 0, 255), img)

    # 6. Localize Poisson Noise Region (left soft tissue area)
    mask_poisson = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask_poisson, (340, 150), 45, 255, -1)
    
    # Generate simulated Poisson-like noise
    noise_amplitude = 20
    poisson_noise = np.random.normal(0, 1, (512, 512)) * np.sqrt(np.maximum(img, 1.0)) * 2.0
    
    img = np.where(mask_poisson == 255, np.clip(img + poisson_noise, 0, 255), img)
    
    # Convert to uint8 and save
    final_img = img.astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, final_img)
    print(f"Generated realistic CT phantom with localized noise regions at: {output_path}")

if __name__ == "__main__":
    create_realistic_ct_phantom("test_ct_image.png")
