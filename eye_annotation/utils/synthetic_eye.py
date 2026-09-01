import numpy as np
import cv2
from typing import Tuple, Dict, Any

def generate_synthetic_eye(
    width: int = 600,
    height: int = 450,
    pupil_r: int = 45,
    iris_r: int = 120,
    pupil_offset: Tuple[int, int] = (0, 0),
    iris_color: Tuple[int, int, int] = (110, 70, 45), # BGR brown/hazel
    glint: bool = True,
    noise_level: float = 0.05,
    blur_level: int = 1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Procedurally generates a realistic synthetic eye image for algorithm testing & demonstration.
    
    Returns:
        image_rgb: np.ndarray (RGB image)
        ground_truth: dict containing exact center & radius of pupil and iris
    """
    center_x = width // 2 + pupil_offset[0]
    center_y = height // 2 + pupil_offset[1]
    
    # 1. Sclera (Background)
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    # Radial falloff for sclera lighting
    yy, xx = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    
    sclera_base = np.array([235, 230, 225], dtype=np.float32) # BGR light pinkish white
    for c in range(3):
        img[:, :, c] = sclera_base[c] - (dist_from_center / np.max(dist_from_center)) * 25.0

    # Add subtle sclera texture/veins
    noise_veins = np.random.normal(0, 8, (height, width))
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] + noise_veins, 0, 255)

    # 2. Iris Disk
    iris_mask = dist_from_center <= iris_r
    
    # Iris gradient & striations
    angles = np.arctan2(yy - center_y, xx - center_x)
    striations = np.sin(angles * 25) * 15.0 + np.cos(angles * 50) * 10.0
    
    iris_norm_dist = dist_from_center / max(1, iris_r)
    iris_radial_gradient = 1.0 - 0.3 * iris_norm_dist
    
    iris_bgr = np.array(iris_color, dtype=np.float32)
    
    for c in range(3):
        channel_val = (iris_bgr[c] + striations) * iris_radial_gradient
        img[:, :, c] = np.where(iris_mask, np.clip(channel_val, 0, 255), img[:, :, c])

    # Dark limbal ring (outer iris boundary edge)
    limbal_mask = (dist_from_center >= (iris_r - 4)) & (dist_from_center <= iris_r)
    for c in range(3):
        img[:, :, c] = np.where(limbal_mask, img[:, :, c] * 0.5, img[:, :, c])

    # 3. Pupil Disk (Dark)
    pupil_mask = dist_from_center <= pupil_r
    pupil_darkness = np.random.normal(18, 4, (height, width)) # Dark grey/black with micro texture
    for c in range(3):
        img[:, :, c] = np.where(pupil_mask, np.clip(pupil_darkness, 5, 45), img[:, :, c])

    # 4. Glint / Specular Reflection
    if glint:
        glint_x = center_x + int(pupil_r * 0.35)
        glint_y = center_y - int(pupil_r * 0.35)
        glint_dist = np.sqrt((xx - glint_x)**2 + (yy - glint_y)**2)
        glint_mask = glint_dist <= max(4, int(pupil_r * 0.2))
        img[glint_mask] = [255, 255, 255]

    # 5. Add Noise & Blur
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, img.shape)
        img = np.clip(img + noise, 0, 255)
        
    img_bgr = img.astype(np.uint8)
    
    if blur_level > 1:
        k = blur_level if blur_level % 2 == 1 else blur_level + 1
        img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    ground_truth = {
        "pupil_center": (center_x, center_y),
        "pupil_radius": pupil_r,
        "iris_center": (center_x, center_y),
        "iris_radius": iris_r
    }
    
    return img_rgb, ground_truth
