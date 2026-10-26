import cv2
import numpy as np
from typing import Dict, Any

def assess_image_quality(gray_img: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates quantitative image quality indicators:
    - Brightness (mean pixel luminance)
    - Contrast (RMS standard deviation)
    - Sharpness (Laplacian variance)
    - Noise Estimate (Mean Absolute Difference)
    - Quality rating & warning text
    """
    if len(gray_img.shape) == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_img.copy()

    # 1. Brightness
    brightness = float(np.mean(gray))

    # 2. Contrast
    contrast = float(np.std(gray))

    # 3. Sharpness (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = max(0.0, float(100.0 - min(100.0, laplacian_var / 5.0)))

    # 4. Noise Estimate (MAD of local difference)
    gray_float = gray.astype(np.float32)
    diff_h = np.abs(gray_float[1:, :] - gray_float[:-1, :])
    diff_w = np.abs(gray_float[:, 1:] - gray_float[:, :-1])
    noise_estimate = float((np.mean(diff_h) + np.mean(diff_w)) / 2.0)

    # Warnings & Quality Level
    is_dark = brightness < 45.0
    is_bright = brightness > 215.0
    is_low_contrast = contrast < 25.0
    is_blurry = laplacian_var < 80.0

    warnings = []
    if is_dark:
        warnings.append("Image is under-exposed / dark.")
    if is_bright:
        warnings.append("Image is over-exposed / bright glare detected.")
    if is_low_contrast:
        warnings.append("Low contrast between pupil/iris and sclera.")
    if is_blurry:
        warnings.append("Image appears blurry / out of focus.")

    if len(warnings) >= 2 or is_blurry or (is_dark and is_low_contrast):
        quality_level = "Low"
        warning_msg = "Low image quality may reduce detection confidence. " + " ".join(warnings)
    elif len(warnings) == 1:
        quality_level = "Medium"
        warning_msg = "Moderate image quality. " + " ".join(warnings)
    else:
        quality_level = "High"
        warning_msg = "Optimal image quality for eye annotation."

    return {
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(laplacian_var, 1),
        "blur_score": round(blur_score, 1),
        "noise_estimate": round(noise_estimate, 1),
        "quality_level": quality_level,
        "warning": warning_msg,
        "is_low_quality": quality_level == "Low"
    }
