"""
Eye Region of Interest (ROI) Detection and Extraction Module.

This module provides functionality to localize, crop, and transform coordinates
for eye regions of interest (ROI) from images. It implements a multi-tier fallback
detection strategy:
1. Primary: OpenCV Haar Cascade eye detector.
2. Fallback: Dark-region & adaptive thresholding around the pupil.
3. Final Fallback: Full image ROI.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class EyeROI:
    """
    Represents a cropped Eye Region of Interest (ROI) and associated spatial metadata.

    Attributes:
        crop: np.ndarray
            The cropped eye image in RGB format (H, W, 3).
        bbox: Tuple[int, int, int, int]
            Bounding box in original image coordinates as (x, y, w, h).
        padding: int
            Amount of padding in pixels added around the raw detection.
        confidence: float
            Confidence score of the ROI localization (0.0 to 100.0).
        method: str
            Detection strategy used: 'haar_cascade', 'dark_region', or 'full_image'.
    """

    crop: np.ndarray
    bbox: Tuple[int, int, int, int]
    padding: int
    confidence: float
    method: str


def _detect_haar(
    gray: np.ndarray,
    img_rgb: np.ndarray,
    padding_factor: float
) -> Optional[EyeROI]:
    """
    Detects an eye ROI using OpenCV Haar cascade classifier.

    Args:
        gray: Grayscale representation of the input image.
        img_rgb: Original RGB image.
        padding_factor: Proportional padding to expand around the detected eye box.

    Returns:
        EyeROI if an eye is detected, or None if detection fails.
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(cascade_path)
        if eye_cascade.empty():
            return None

        # Detect eyes in grayscale image
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        if len(eyes) == 0:
            return None

        # Select the largest detection by bounding box area (w * h)
        x, y, w, h = max(eyes, key=lambda box: int(box[2]) * int(box[3]))
        eye_size = max(int(w), int(h))
        pad = int(round(padding_factor * eye_size))

        img_h, img_w = img_rgb.shape[:2]

        # Apply padding and clamp to image bounds
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(img_w, int(x) + int(w) + pad)
        y2 = min(img_h, int(y) + int(h) + pad)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 10 or crop_h <= 10:
            return None

        crop = img_rgb[y1:y2, x1:x2].copy()
        bbox = (int(x1), int(y1), int(crop_w), int(crop_h))

        return EyeROI(
            crop=crop,
            bbox=bbox,
            padding=pad,
            confidence=90.0,
            method="haar_cascade"
        )
    except Exception:
        return None


def _detect_dark_region(
    gray: np.ndarray,
    img_rgb: np.ndarray,
    padding_factor: float
) -> Optional[EyeROI]:
    """
    Detects an eye ROI by finding the primary dark blob (pupil/iris area)
    via adaptive thresholding and intensity masking.

    Args:
        gray: Grayscale representation of the input image.
        img_rgb: Original RGB image.
        padding_factor: Proportional padding to expand around the estimated eye region.

    Returns:
        EyeROI centered around the dark region, or None if no blob is found.
    """
    try:
        img_h, img_w = img_rgb.shape[:2]

        # Blur to suppress noise and small specular reflections
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive thresholding to extract locally dark structures
        block_size = max(15, (min(img_h, img_w) // 8) | 1)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            5
        )

        # Intensity masking: pupil is among the darkest regions globally
        p35 = int(np.percentile(blurred, 35))
        dark_cutoff = max(p35, 45)
        dark_mask = (blurred <= dark_cutoff).astype(np.uint8) * 255
        combined_binary = cv2.bitwise_and(adaptive_thresh, dark_mask)

        # Morphological opening to eliminate isolated small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel)

        # Find candidate contours
        contours, _ = cv2.findContours(
            cleaned_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # If opening removed everything, fall back to raw adaptive thresholding
        if not contours:
            contours, _ = cv2.findContours(
                adaptive_thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

        # If still empty, fall back to global percentile thresholding
        if not contours:
            p15 = int(np.percentile(blurred, 15))
            _, p_thresh = cv2.threshold(blurred, max(p15, 30), 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(
                p_thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

        min_blob_area = 15.0
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_blob_area]

        if not valid_contours:
            return None

        # Select the largest dark blob
        largest_blob = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_blob)

        # Determine centroid
        m = cv2.moments(largest_blob)
        if m["m00"] > 0:
            cx = int(round(m["m10"] / m["m00"]))
            cy = int(round(m["m01"] / m["m00"]))
        else:
            (cx_f, cy_f), _ = cv2.minEnclosingCircle(largest_blob)
            cx, cy = int(round(cx_f)), int(round(cy_f))

        # Equivalent circular radius of the dark blob
        radius = np.sqrt(area / np.pi)
        radius = max(radius, min(img_h, img_w) * 0.02)

        # Expand outward by a factor proportional to blob radius (5x blob radius)
        base_half_size = int(round(5.0 * radius))
        pad = int(round(padding_factor * (2 * base_half_size)))
        half_size = base_half_size + pad

        # Ensure reasonable bounding scale
        half_size = max(half_size, int(min(img_h, img_w) * 0.12))

        # Clamp to image bounds
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(img_w, cx + half_size)
        y2 = min(img_h, cy + half_size)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 10 or crop_h <= 10:
            return None

        crop = img_rgb[y1:y2, x1:x2].copy()
        bbox = (int(x1), int(y1), int(crop_w), int(crop_h))

        return EyeROI(
            crop=crop,
            bbox=bbox,
            padding=pad,
            confidence=70.0,
            method="dark_region"
        )
    except Exception:
        return None


def _detect_full_image(img_rgb: np.ndarray) -> EyeROI:
    """
    Fallback method using the entire image as the Eye ROI.

    Args:
        img_rgb: Original RGB image.

    Returns:
        EyeROI spanning the complete image canvas.
    """
    img_h, img_w = img_rgb.shape[:2]
    return EyeROI(
        crop=img_rgb.copy(),
        bbox=(0, 0, int(img_w), int(img_h)),
        padding=0,
        confidence=30.0,
        method="full_image"
    )


def detect_eye_roi(
    image_rgb: np.ndarray,
    padding_factor: float = 0.3
) -> EyeROI:
    """
    Detects and crops the eye region of interest (ROI) from a full image.

    The detection strategy proceeds through three progressive stages:
    1. Primary: OpenCV Haar cascade eye detector ('haar_cascade').
    2. Fallback: Dark-region centroid expansion around the pupil ('dark_region').
    3. Final Fallback: Entire image canvas ('full_image').

    Args:
        image_rgb: Input image as a numpy ndarray (RGB or grayscale).
        padding_factor: Fractional margin added around the detected eye region
            (default 0.3).

    Returns:
        EyeROI dataclass instance containing the cropped RGB array,
        original bounding box (x, y, w, h), padding pixels, confidence,
        and the detection method employed.

    Raises:
        ValueError: If image_rgb is None, not a numpy array, or empty.
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        raise ValueError("Input image_rgb must be a non-empty numpy ndarray.")

    # Normalize color channels to 3-channel RGB
    if image_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        img_rgb = image_rgb.copy()
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
        img_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {image_rgb.shape}")

    padding_factor = max(0.0, float(padding_factor))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Method 1 (primary): Haar Cascade
    roi = _detect_haar(gray, img_rgb, padding_factor)
    if roi is not None:
        return roi

    # Method 2 (fallback): Dark-Region Adaptive Thresholding
    roi = _detect_dark_region(gray, img_rgb, padding_factor)
    if roi is not None:
        return roi

    # Method 3 (final fallback): Full Image
    return _detect_full_image(img_rgb)


def roi_to_full_coords(roi: EyeROI, point: Tuple[int, int]) -> Tuple[int, int]:
    """
    Maps a coordinate point from ROI-local coordinates back to full image coordinates.

    Args:
        roi: EyeROI instance containing the bounding box (x, y, w, h).
        point: Tuple (x_roi, y_roi) in ROI-local pixel coordinates.

    Returns:
        Tuple (x_full, y_full) in full image coordinates.
    """
    x_full = int(round(point[0] + roi.bbox[0]))
    y_full = int(round(point[1] + roi.bbox[1]))
    return (x_full, y_full)


def full_to_roi_coords(roi: EyeROI, point: Tuple[int, int]) -> Tuple[int, int]:
    """
    Maps a coordinate point from full image coordinates to ROI-local coordinates.

    Args:
        roi: EyeROI instance containing the bounding box (x, y, w, h).
        point: Tuple (x_full, y_full) in full image coordinates.

    Returns:
        Tuple (x_roi, y_roi) in ROI-local pixel coordinates.
    """
    x_roi = int(round(point[0] - roi.bbox[0]))
    y_roi = int(round(point[1] - roi.bbox[1]))
    return (x_roi, y_roi)
