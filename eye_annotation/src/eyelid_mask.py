"""
Eyelid and eyelash detection and masking.

Detects upper and lower eyelid curves using edge detection and polynomial
curve fitting. Generates a binary mask where 1 = valid iris search region,
0 = occluded by eyelid/eyelash.

This is critical for preventing eyelid edges from being mistakenly
identified as iris boundaries.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class EyelidMask:
    """Stores eyelid detection results and the valid-region mask."""

    def __init__(
        self,
        valid_mask: Optional[np.ndarray] = None,
        upper_curve_y: Optional[np.ndarray] = None,
        lower_curve_y: Optional[np.ndarray] = None,
        occlusion_pct: float = 0.0,
        detected: bool = False
    ):
        self.valid_mask = valid_mask
        self.upper_curve_y = upper_curve_y   # y-value per x column for upper eyelid
        self.lower_curve_y = lower_curve_y   # y-value per x column for lower eyelid
        self.occlusion_pct = occlusion_pct
        self.detected = detected


def detect_eyelid_mask(
    gray: np.ndarray,
    pupil_center: Tuple[int, int],
    pupil_radius: float,
    search_radius: float = 0.0
) -> EyelidMask:
    """
    Detects upper and lower eyelid boundaries and generates a mask
    of the region valid for iris boundary search.

    Parameters
    ----------
    gray : grayscale image (uint8)
    pupil_center : (x, y) of detected pupil center
    pupil_radius : pupil radius in pixels
    search_radius : if > 0, the expected iris search radius (used to limit mask region)

    Returns
    -------
    EyelidMask with valid_mask (same size as gray), curve data, and occlusion percentage.
    """
    h, w = gray.shape[:2]
    px, py = int(pupil_center[0]), int(pupil_center[1])
    pr = max(5.0, pupil_radius)

    if search_radius <= 0:
        search_radius = pr * 3.5

    # Create the full valid mask (start with all valid)
    valid_mask = np.ones((h, w), dtype=np.uint8)

    # Define the search region for eyelid edges
    margin = int(search_radius * 1.2)
    x_min = max(0, px - margin)
    x_max = min(w, px + margin)
    y_min_upper = max(0, py - margin)
    y_max_lower = min(h, py + margin)

    # --- Upper eyelid detection ---
    upper_curve_y = _detect_eyelid_curve(
        gray, px, py, pr, x_min, x_max,
        y_min_upper, py,  # search above pupil center
        is_upper=True
    )

    # --- Lower eyelid detection ---
    lower_curve_y = _detect_eyelid_curve(
        gray, px, py, pr, x_min, x_max,
        py, y_max_lower,  # search below pupil center
        is_upper=False
    )

    # Build the mask: invalid above upper eyelid and below lower eyelid
    if upper_curve_y is not None:
        for x in range(x_min, x_max):
            upper_y = int(round(upper_curve_y[x - x_min]))
            upper_y = max(0, min(h - 1, upper_y))
            valid_mask[0:upper_y, x] = 0

    if lower_curve_y is not None:
        for x in range(x_min, x_max):
            lower_y = int(round(lower_curve_y[x - x_min]))
            lower_y = max(0, min(h - 1, lower_y))
            valid_mask[lower_y:h, x] = 0

    # Also mask strong eyelash edges near the eyelid curves
    _mask_eyelash_edges(gray, valid_mask, px, py, pr, search_radius)

    # Compute occlusion percentage within the iris search annulus
    iris_annulus = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_annulus, (px, py), int(search_radius), 255, -1)
    cv2.circle(iris_annulus, (px, py), int(pr * 1.2), 0, -1)
    annulus_pixels = np.sum(iris_annulus > 0)
    if annulus_pixels > 0:
        occluded_pixels = np.sum((iris_annulus > 0) & (valid_mask == 0))
        occlusion_pct = (occluded_pixels / float(annulus_pixels)) * 100.0
    else:
        occlusion_pct = 0.0

    return EyelidMask(
        valid_mask=valid_mask,
        upper_curve_y=upper_curve_y,
        lower_curve_y=lower_curve_y,
        occlusion_pct=round(occlusion_pct, 1),
        detected=(upper_curve_y is not None or lower_curve_y is not None)
    )


def _detect_eyelid_curve(
    gray: np.ndarray,
    px: int, py: int, pr: float,
    x_min: int, x_max: int,
    y_search_min: int, y_search_max: int,
    is_upper: bool
) -> Optional[np.ndarray]:
    """
    Detects one eyelid curve (upper or lower) by finding strong horizontal
    edges and fitting a parabolic curve.

    Returns array of y-values for each x in [x_min, x_max), or None if detection fails.
    """
    h, w = gray.shape[:2]
    region_w = x_max - x_min

    if region_w < 10 or (y_search_max - y_search_min) < 5:
        return None

    # Extract the search region
    roi = gray[y_search_min:y_search_max, x_min:x_max]
    if roi.size == 0:
        return None

    # Compute horizontal edge response (Sobel Y)
    # Upper eyelid: look for dark-above-light-below transition (negative Sobel Y)
    # Lower eyelid: look for light-above-dark-below transition (positive Sobel Y)
    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 1.5)
    sobel_y = cv2.Sobel(blurred_roi, cv2.CV_64F, 0, 1, ksize=3)

    if is_upper:
        # Upper eyelid: the edge is where skin/eyelid meets the eye opening
        # This is typically a strong negative-to-positive gradient transition
        edge_response = np.abs(sobel_y)
    else:
        edge_response = np.abs(sobel_y)

    # Smooth horizontally to get continuous curves
    edge_smooth = cv2.GaussianBlur(edge_response, (15, 3), 3.0)

    # For each column, find the strongest edge
    roi_h, roi_w = edge_smooth.shape
    edge_points_x = []
    edge_points_y = []

    # Sample columns with some spacing
    step = max(1, region_w // 30)
    for col_offset in range(0, roi_w, step):
        col_data = edge_smooth[:, col_offset]
        if len(col_data) < 3:
            continue

        # Find the strongest edge in this column
        peak_idx = int(np.argmax(col_data))
        peak_val = col_data[peak_idx]

        # Threshold: edge must be reasonably strong
        if peak_val < 10.0:
            continue

        # Map back to image coordinates
        edge_x = x_min + col_offset
        edge_y = y_search_min + peak_idx

        # Sanity check: edge should be outside the pupil
        dist_from_pupil = np.sqrt((edge_x - px) ** 2 + (edge_y - py) ** 2)
        if dist_from_pupil < pr * 1.2:
            continue

        edge_points_x.append(edge_x)
        edge_points_y.append(edge_y)

    if len(edge_points_x) < 5:
        return None

    edge_points_x = np.array(edge_points_x, dtype=np.float64)
    edge_points_y = np.array(edge_points_y, dtype=np.float64)

    # Fit a 2nd-degree polynomial (parabola) to the edge points
    # Use RANSAC-like approach: fit, remove outliers, refit
    try:
        coeffs = np.polyfit(edge_points_x, edge_points_y, 2)
    except Exception:
        return None

    # Remove outliers (points far from the fitted curve)
    fitted_y = np.polyval(coeffs, edge_points_x)
    residuals = np.abs(edge_points_y - fitted_y)
    median_residual = float(np.median(residuals))
    inlier_mask = residuals < max(5.0, median_residual * 2.5)

    if np.sum(inlier_mask) < 4:
        return None

    # Refit on inliers
    try:
        coeffs = np.polyfit(edge_points_x[inlier_mask], edge_points_y[inlier_mask], 2)
    except Exception:
        return None

    # Validate: for upper eyelid, the parabola should open downward (concave, a < 0)
    # For lower eyelid, it should open upward (convex, a > 0)
    a_coeff = coeffs[0]
    if is_upper and a_coeff > 0.001:
        # Upper eyelid curve opens upward — wrong shape, probably not an eyelid
        # Still use it but with reduced confidence
        pass
    if not is_upper and a_coeff < -0.001:
        pass

    # Generate the curve for all x values in the region
    x_range = np.arange(x_min, x_max, dtype=np.float64)
    curve_y = np.polyval(coeffs, x_range)

    # Clamp to image bounds
    curve_y = np.clip(curve_y, 0, h - 1)

    return curve_y


def _mask_eyelash_edges(
    gray: np.ndarray,
    valid_mask: np.ndarray,
    px: int, py: int, pr: float,
    search_radius: float
):
    """
    Detects and masks strong eyelash-like edges near the iris region.
    Eyelashes appear as thin, high-contrast, roughly vertical/diagonal lines.
    """
    h, w = gray.shape[:2]

    # Search in the region around the iris
    margin = int(search_radius * 1.0)
    y1 = max(0, py - margin)
    y2 = min(h, py + margin)
    x1 = max(0, px - margin)
    x2 = min(w, px + margin)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # Detect thin dark lines using morphological black-hat transform
    # (finds dark structures smaller than the structuring element)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel_v)

    # Threshold the blackhat to find strong eyelash-like structures
    _, lash_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

    # Dilate slightly to cover the edge influence of eyelashes
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lash_mask = cv2.dilate(lash_mask, kernel_dilate, iterations=1)

    # Apply to the valid mask (mask out eyelash regions)
    valid_mask[y1:y2, x1:x2] = np.where(lash_mask > 0, 0, valid_mask[y1:y2, x1:x2])
