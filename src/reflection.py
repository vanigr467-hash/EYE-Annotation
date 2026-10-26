"""
Specular Glint and Reflection Detection and Masking Module.

Identifies bright specular reflection highlights (glints) on the corneal surface
using adaptive intensity thresholding and morphological cleanup.
Provides non-destructive reflection masks and optional localized inpainting.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


def detect_specular_reflections(
    image_gray: np.ndarray,
    threshold_percentile: float = 98.0,
    min_intensity: int = 220,
    dilation_kernel_size: int = 5
) -> Dict[str, Any]:
    """
    Detects specular reflection hotspots and returns a clean binary mask and diagnostics.
    
    Args:
        image_gray: 2D uint8 grayscale eye image
        threshold_percentile: percentile for adaptive bright hotspot identification
        min_intensity: absolute minimum pixel intensity to be considered a glint
        dilation_kernel_size: size of morphological dilation kernel
        
    Returns:
        dict with:
            'mask': 2D uint8 binary mask (255 = reflection, 0 = normal)
            'glint_percentage': float percentage of pixels flagged
            'num_glints': int number of distinct glint connected components
            'glint_centers': List[Tuple[int, int]] centroids of glints
    """
    h, w = image_gray.shape[:2]

    # Calculate adaptive threshold
    p_thresh = float(np.percentile(image_gray, threshold_percentile))
    active_thresh = int(max(min_intensity, p_thresh))

    # Binary threshold for bright spots
    _, binary_glints = cv2.threshold(image_gray, active_thresh, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
    )
    glint_mask = cv2.dilate(binary_glints, kernel, iterations=1)

    # Find connected components for diagnostics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(glint_mask)
    
    glint_centers = []
    # label 0 is background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 3:  # ignore 1-2 pixel isolated noise
            cx, cy = int(round(centroids[i][0])), int(round(centroids[i][1]))
            glint_centers.append((cx, cy))

    glint_pixels = int(np.sum(glint_mask > 0))
    glint_pct = round((glint_pixels / float(h * w)) * 100.0, 2)

    return {
        "mask": glint_mask,
        "glint_percentage": glint_pct,
        "num_glints": len(glint_centers),
        "glint_centers": glint_centers
    }


def inpaint_reflections(
    image_gray: np.ndarray,
    glint_mask: np.ndarray,
    inpaint_radius: int = 5,
    method: str = "telea"
) -> np.ndarray:
    """
    Inpaints specular reflections cleanly using Telea or Navier-Stokes.
    Used internally for edge/candidate analysis without altering the raw user image.
    """
    if np.sum(glint_mask) == 0:
        return image_gray.copy()

    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(image_gray, glint_mask, inpaint_radius, inpaint_flag)
    return inpainted
