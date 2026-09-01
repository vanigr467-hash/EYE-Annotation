import cv2
import numpy as np
from typing import Dict, Any

def preprocess_eye_image(
    image_rgb: np.ndarray,
    clahe_clip: float = 2.0,
    blur_kernel: int = 5,
    glint_removal: bool = True
) -> Dict[str, Any]:
    """
    Applies comprehensive image preprocessing for pupil and iris detection.
    
    Steps:
    1. RGB to Grayscale
    2. Specular Glint Inpainting (morphological closure inside dark regions)
    3. Contrast enhancement via CLAHE
    4. Gaussian & Bilateral Noise Filtering
    
    Returns dict with intermediate preprocessed images.
    """
    # 1. Grayscale conversion
    if len(image_rgb.shape) == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.copy()

    # 2. Specular Reflection / Glint Inpainting
    glint_mask = np.zeros_like(gray)
    if glint_removal:
        # High intensity spots (> 220) surrounded by dark regions
        _, bright_spots = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        kernel_glint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glint_mask = cv2.dilate(bright_spots, kernel_glint, iterations=2)
        
        # Inpaint glint spots using Navier-Stokes/Telea inpainting
        gray_inpainted = cv2.inpaint(gray, glint_mask, 5, cv2.INPAINT_TELEA)
    else:
        gray_inpainted = gray.copy()

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_inpainted)

    # 4. Noise Reduction / Smoothing
    k_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    gray_blurred = cv2.GaussianBlur(gray_clahe, (k_size, k_size), 0)
    
    # Bilateral filter for edge-preserving smoothing
    gray_bilateral = cv2.bilateralFilter(gray_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    return {
        "original_rgb": image_rgb,
        "gray": gray,
        "glint_mask": glint_mask,
        "gray_inpainted": gray_inpainted,
        "clahe": gray_clahe,
        "blurred": gray_blurred,
        "bilateral": gray_bilateral
    }
