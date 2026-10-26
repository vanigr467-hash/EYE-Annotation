"""
Image Quality Assessment Module for Eye Annotation System.

This module provides diagnostic quality assessment for eye images and video frames.
It computes quantitative metrics including resolution, exposure, contrast, sharpness,
noise estimation via Laplacian Median Absolute Deviation (MAD), dynamic range,
and composite quality scoring with grade categorization.
"""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def analyze_image_quality(gray: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes quantitative image quality indicators for an eye image or video frame.

    Evaluates:
    - Resolution and pixel count
    - Brightness and exposure distribution (overexposure, underexposure, specular highlights)
    - Contrast (RMS standard deviation of pixel intensities)
    - Sharpness (Laplacian variance)
    - Noise estimate (Median Absolute Deviation of the Laplacian)
    - Dynamic range (95th percentile - 5th percentile)
    - Composite quality score (0-100) and grade (EXCELLENT, GOOD, FAIR, LOW, VERY_LOW)
    - Diagnostic warnings for quality defects

    Args:
        gray: Input image as a 2D grayscale NumPy array, or 3D BGR/RGB image.

    Returns:
        A dictionary containing diagnostic metrics, categories, composite score,
        grade, low-quality flag, and backward-compatible fields.

    Raises:
        ValueError: If input image is empty or has unsupported dimensions.
    """
    if gray is None:
        raise ValueError("Input image cannot be None.")

    if not isinstance(gray, np.ndarray):
        raise TypeError(f"Input image must be a numpy.ndarray, got {type(gray).__name__}.")

    if gray.size == 0:
        raise ValueError("Input image array is empty.")

    # Convert 3D RGB/BGR images to 2D grayscale if necessary
    if gray.ndim == 3:
        channels = gray.shape[2]
        if channels == 3:
            gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray_img = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            gray_img = gray[:, :, 0].copy()
        else:
            raise ValueError(f"Unsupported number of image channels: {channels}")
    elif gray.ndim == 2:
        gray_img = gray.copy()
    else:
        raise ValueError(f"Input image must have 2 or 3 dimensions, got shape {gray.shape}.")

    # Ensure 8-bit unsigned integer or single-channel float for uniform calculation
    if gray_img.dtype != np.uint8:
        if np.issubdtype(gray_img.dtype, np.floating):
            if gray_img.max() <= 1.0 and gray_img.min() >= 0.0:
                gray_img = np.clip(gray_img * 255.0, 0, 255).astype(np.uint8)
            else:
                gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
        else:
            gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)

    height, width = gray_img.shape[:2]
    resolution: Tuple[int, int] = (int(width), int(height))
    total_pixels: int = int(width * height)

    # 1. Mean brightness (0-255)
    mean_brightness: float = float(np.mean(gray_img))

    # 2. Exposure metrics & specular reflection
    overexposure_count: int = int(np.count_nonzero(gray_img > 240))
    underexposure_count: int = int(np.count_nonzero(gray_img < 15))
    specular_count: int = int(np.count_nonzero(gray_img > 250))

    overexposure_pct: float = float((overexposure_count / total_pixels) * 100.0)
    underexposure_pct: float = float((underexposure_count / total_pixels) * 100.0)
    specular_pct: float = float((specular_count / total_pixels) * 100.0)

    # Brightness categorization
    if mean_brightness >= 200.0 or overexposure_pct > 25.0:
        brightness_category: str = "OVEREXPOSED"
    elif mean_brightness < 60.0 or underexposure_pct > 35.0:
        brightness_category: str = "DARK"
    elif mean_brightness > 165.0:
        brightness_category: str = "BRIGHT"
    else:
        brightness_category: str = "NORMAL"

    # 3. Contrast (RMS standard deviation)
    contrast: float = float(np.std(gray_img))
    if contrast < 20.0:
        contrast_category: str = "LOW"
    elif contrast > 55.0:
        contrast_category: str = "HIGH"
    else:
        contrast_category: str = "NORMAL"

    # 4. Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    sharpness: float = float(laplacian.var())

    if sharpness < 50.0:
        sharpness_category: str = "VERY_BLURRY"
    elif sharpness < 150.0:
        sharpness_category: str = "BLURRY"
    elif sharpness < 450.0:
        sharpness_category: str = "ACCEPTABLE"
    else:
        sharpness_category: str = "SHARP"

    # 5. Noise estimate (Median Absolute Deviation of Laplacian)
    # MAD = median(|Laplacian - median(Laplacian)|)
    lap_median = float(np.median(laplacian))
    noise_estimate: float = float(np.median(np.abs(laplacian - lap_median)))

    # 6. Dynamic Range (95th percentile - 5th percentile)
    p95, p5 = np.percentile(gray_img, [95, 5])
    dynamic_range: float = float(p95 - p5)

    # 7. Quality Score Computation (0-100 composite)
    # Sharpness score (0-25): >500 = 25, <50 = 0
    if sharpness <= 50.0:
        sharpness_score: float = 0.0
    elif sharpness >= 500.0:
        sharpness_score = 25.0
    else:
        sharpness_score = 25.0 * (sharpness - 50.0) / (500.0 - 50.0)

    # Contrast score (0-25): >50 = 25, <15 = 0
    if contrast <= 15.0:
        contrast_score: float = 0.0
    elif contrast >= 50.0:
        contrast_score = 25.0
    else:
        contrast_score = 25.0 * (contrast - 15.0) / (50.0 - 15.0)

    # Brightness score (0-25): penalty for over/underexposure
    # Deviation from ideal mean brightness range [80, 160]
    if mean_brightness < 80.0:
        mean_dev_penalty = ((80.0 - mean_brightness) / 80.0) * 12.5
    elif mean_brightness > 160.0:
        mean_dev_penalty = ((mean_brightness - 160.0) / 95.0) * 12.5
    else:
        mean_dev_penalty = 0.0

    under_penalty = min(6.25, (underexposure_pct / 20.0) * 6.25)
    over_penalty = min(6.25, ((overexposure_pct + specular_pct) / 20.0) * 6.25)
    total_brightness_penalty = mean_dev_penalty + under_penalty + over_penalty
    brightness_score: float = max(0.0, min(25.0, 25.0 - total_brightness_penalty))

    # Noise score (0-25): lower noise = higher score
    # Laplacian MAD: <= 2.0 -> 25, >= 20.0 -> 0
    if noise_estimate <= 2.0:
        noise_score: float = 25.0
    elif noise_estimate >= 20.0:
        noise_score = 0.0
    else:
        noise_score = 25.0 * (1.0 - (noise_estimate - 2.0) / 18.0)

    composite_score: float = float(
        np.clip(sharpness_score + contrast_score + brightness_score + noise_score, 0.0, 100.0)
    )

    # 8. Quality Grade Categorization
    if composite_score >= 80.0:
        quality_grade: str = "EXCELLENT"
    elif composite_score >= 65.0:
        quality_grade = "GOOD"
    elif composite_score >= 50.0:
        quality_grade = "FAIR"
    elif composite_score >= 35.0:
        quality_grade = "LOW"
    else:
        quality_grade = "VERY_LOW"

    # Low quality flag: triggered if low/very_low grade or severely blurry
    is_low_quality: bool = bool(quality_grade in ("LOW", "VERY_LOW") or sharpness_category == "VERY_BLURRY")

    # 9. Diagnostic Warnings List
    warnings: List[str] = []
    if sharpness_category == "VERY_BLURRY":
        warnings.append("Image is severely blurry / out of focus (Laplacian variance < 50).")
    elif sharpness_category == "BLURRY":
        warnings.append("Image appears blurry; fine boundary details may be degraded.")

    if contrast_category == "LOW":
        warnings.append("Low contrast between pupil/iris and sclera (std dev < 20).")

    if brightness_category == "DARK" or underexposure_pct > 25.0:
        warnings.append(f"Image is under-exposed / dark ({underexposure_pct:.1f}% pixels < 15).")
    elif brightness_category == "OVEREXPOSED" or overexposure_pct > 20.0:
        warnings.append(f"Image is over-exposed / bright glare detected ({overexposure_pct:.1f}% pixels > 240).")

    if specular_pct > 5.0:
        warnings.append(f"High specular reflection / corneal glare ({specular_pct:.1f}% pixels > 250).")

    if noise_estimate > 12.0:
        warnings.append(f"Elevated image noise detected (Laplacian MAD: {noise_estimate:.1f}).")

    if dynamic_range < 50.0:
        warnings.append(f"Narrow dynamic range ({dynamic_range:.1f} intensity levels).")

    # 10. Backward Compatibility Fields
    quality_level: str = quality_grade
    first_warning: str = warnings[0] if warnings else ""
    legacy_blur_score: float = max(0.0, float(100.0 - min(100.0, sharpness / 5.0)))

    return {
        # Core requested metrics
        "resolution": resolution,
        "total_pixels": total_pixels,
        "mean_brightness": round(mean_brightness, 2),
        "brightness_category": brightness_category,
        "contrast": round(contrast, 2),
        "contrast_category": contrast_category,
        "sharpness": round(sharpness, 2),
        "sharpness_category": sharpness_category,
        "noise_estimate": round(noise_estimate, 2),
        "overexposure_pct": round(overexposure_pct, 2),
        "underexposure_pct": round(underexposure_pct, 2),
        "specular_pct": round(specular_pct, 2),
        "dynamic_range": round(dynamic_range, 2),
        "quality_grade": quality_grade,
        "quality_score": round(composite_score, 1),
        "is_low_quality": is_low_quality,
        "warnings": warnings,
        # Component scores
        "sharpness_score": round(sharpness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "brightness_score": round(brightness_score, 2),
        "noise_score": round(noise_score, 2),
        # Backward compatibility fields matching quality.py
        "quality_level": quality_level,
        "warning": first_warning,
        "brightness": round(mean_brightness, 1),
        "blur_score": round(legacy_blur_score, 1),
    }


def assess_image_quality(gray_img: np.ndarray) -> Dict[str, Any]:
    """
    Backward-compatible entry point for legacy code calling assess_image_quality.

    Args:
        gray_img: Input grayscale or color NumPy image array.

    Returns:
        Comprehensive quality assessment metrics dictionary.
    """
    return analyze_image_quality(gray_img)
