"""
Multi-Stage Preprocessing Ensemble Module.

Generates 4 distinct preprocessing variants and automatically selects the optimal 
version for pupil/iris boundary detection based on local contrast and signal quality:
- Variant A: CLAHE + Bilateral edge-preserving denoising
- Variant B: Adaptive Gamma Correction + CLAHE
- Variant C: Local Contrast (Top-Hat / Bottom-Hat) + Mild Sharpening
- Variant D: Illumination Normalization (Homomorphic-style division)
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from src.image_quality import analyze_image_quality
from src.reflection import detect_specular_reflections, inpaint_reflections


def preprocess_eye_image(
    image_rgb: np.ndarray,
    clahe_clip: float = 2.0,
    blur_kernel: int = 5,
    glint_removal: bool = True,
    gamma: float = 1.0,
    force_variant: str = "auto"
) -> Dict[str, Any]:
    """
    Runs the multi-variant preprocessing pipeline and returns intermediate 
    and selected representations along with quality diagnostics.
    
    Args:
        image_rgb: RGB input image (H, W, 3) or 2D Grayscale
        clahe_clip: CLAHE clip limit (default 2.0)
        blur_kernel: kernel size for noise reduction
        glint_removal: whether to detect and inpaint specular glints
        gamma: manual gamma correction override (1.0 = automatic)
        force_variant: 'auto', 'A', 'B', 'C', or 'D'
        
    Returns:
        Dictionary containing all variants, selected best, quality diagnostics, and glint mask.
    """
    # 1. Grayscale conversion
    if len(image_rgb.shape) == 3:
        gray_raw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray_raw = image_rgb.copy()

    # 2. Quality analysis
    quality_info = analyze_image_quality(gray_raw)

    # 3. Specular Glint Handling
    glint_info = detect_specular_reflections(gray_raw)
    glint_mask = glint_info["mask"]

    if glint_removal and (glint_info["glint_percentage"] > 0.0 or np.any(glint_mask > 0)):
        gray_inpainted = inpaint_reflections(gray_raw, glint_mask, inpaint_radius=4)
    else:
        gray_inpainted = gray_raw.copy()

    # Pure edge-preserving bilateral filter on raw inpainted image (NO histogram equalization)
    # This preserves natural pupil-to-iris and iris-to-sclera gradients without amplifying eyelash texture
    bilateral_raw = cv2.bilateralFilter(gray_inpainted, d=7, sigmaColor=35, sigmaSpace=35)

    # 4. Generate Preprocessing Ensemble Variants

    # --- Variant A: Standard CLAHE + Bilateral Filtering ---
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    var_a_clahe = clahe.apply(gray_inpainted)
    var_a_bilateral = cv2.bilateralFilter(var_a_clahe, d=7, sigmaColor=50, sigmaSpace=50)

    # --- Variant B: Adaptive Gamma Correction + CLAHE ---
    mean_b = quality_info["mean_brightness"]
    # Auto-adjust gamma if image is under/overexposed
    if gamma == 1.0:
        if mean_b < 80:
            eff_gamma = 0.75  # Brighten shadows
        elif mean_b > 160:
            eff_gamma = 1.35  # Darken highlights
        else:
            eff_gamma = 1.0
    else:
        eff_gamma = gamma

    if eff_gamma != 1.0:
        inv_gamma = 1.0 / eff_gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray_gamma = cv2.LUT(gray_inpainted, table)
    else:
        gray_gamma = gray_inpainted.copy()

    clahe_b = cv2.createCLAHE(clipLimit=max(1.5, clahe_clip * 1.2), tileGridSize=(8, 8))
    var_b = clahe_b.apply(gray_gamma)

    # --- Variant C: Morphological Contrast Enhancement + Mild Sharpening ---
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray_inpainted, cv2.MORPH_TOPHAT, kernel_morph)
    blackhat = cv2.morphologyEx(gray_inpainted, cv2.MORPH_BLACKHAT, kernel_morph)
    enhanced = cv2.add(gray_inpainted, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    
    # Mild unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    var_c = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)

    # --- Variant D: Illumination Normalization ---
    # Divide image by large low-pass background estimate to cancel uneven lighting
    bg = cv2.GaussianBlur(gray_inpainted, (41, 41), 0)
    bg = np.maximum(bg, 1).astype(np.float32)
    normalized = (gray_inpainted.astype(np.float32) / bg) * 128.0
    var_d = np.clip(normalized, 0, 255).astype(np.uint8)
    var_d = cv2.equalizeHist(var_d)

    # 5. Automatic Variant Selection Score
    def compute_selection_score(img_variant: np.ndarray) -> float:
        # Measure local gradient energy in the mid-frequency band
        sobel_x = cv2.Sobel(img_variant, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_variant, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # We want high edge strength but low noise (measured by std dev of smooth regions)
        p10 = np.percentile(img_variant, 10)
        p90 = np.percentile(img_variant, 90)
        dynamic_range = p90 - p10
        edge_energy = np.mean(mag)
        
        return float(dynamic_range * 0.6 + edge_energy * 0.4)

    scores = {
        "A": compute_selection_score(var_a_bilateral),
        "B": compute_selection_score(var_b),
        "C": compute_selection_score(var_c),
        "D": compute_selection_score(var_d)
    }

    if force_variant.upper() in ["A", "B", "C", "D"]:
        selected_key = force_variant.upper()
    else:
        selected_key = max(scores, key=scores.get)

    variant_map = {
        "A": var_a_bilateral,
        "B": var_b,
        "C": var_c,
        "D": var_d
    }
    selected_img = variant_map[selected_key]

    # Noise reduction kernel
    k_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    selected_blurred = cv2.GaussianBlur(selected_img, (k_size, k_size), 0)

    return {
        "original_rgb": image_rgb,
        "gray": gray_raw,
        "glint_mask": glint_mask,
        "glint_info": glint_info,
        "gray_inpainted": gray_inpainted,
        "gray_gamma": gray_gamma,
        "clahe": var_a_clahe,
        "blurred": selected_blurred,
        "bilateral": var_a_bilateral,
        "bilateral_raw": bilateral_raw,
        "variant_a": var_a_bilateral,
        "variant_b": var_b,
        "variant_c": var_c,
        "variant_d": var_d,
        "variant_scores": scores,
        "selected_variant": selected_key,
        "processed_primary": selected_img,
        "quality": quality_info
    }
