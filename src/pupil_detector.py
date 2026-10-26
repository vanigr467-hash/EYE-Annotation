"""
Multi-Candidate Hybrid Pupil Detection and Ellipse Fitting Module.

Combines multiple candidate extraction strategies focused on true dark pupil regions:
1. Glint-Compensated Morphological Hole Filling
2. Adaptive Low-Percentile Dark Region Thresholding
3. Complete Contour Hierarchy Analysis (RETR_LIST) with Solid-Disk Verification
4. Circularity, Local Contrast, and Relative Darkness Scoring
5. Circular Hough Transform with Relative Darkness Validation
6. Robust RANSAC Ellipse Fitting on Inlier Boundary Points
7. Starburst Sub-Pixel Radial Ray Refinement
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import cv2
import numpy as np

from src.config_constants import (
    CANDIDATE_PERCENTILES,
    PUPIL_MIN_RADIUS_RATIO,
    PUPIL_MAX_RADIUS_RATIO,
    PUPIL_MIN_AXIS_RATIO,
    PUPIL_BORDER_MARGIN_RATIO
)
from src.candidate_scoring import score_pupil_candidate
from src.ellipse_fitting import ransac_ellipse_fit
from src.starburst_refinement import refine_pupil_boundary_starburst
from src.confidence import compute_calibrated_confidence
from src.plausibility import validate_detection


class PupilResult:
    """Stores pupil detection results and quality diagnostics."""

    def __init__(
        self,
        center: Tuple[int, int] = (0, 0),
        radius: float = 0.0,
        radius_x: float = 0.0,
        radius_y: float = 0.0,
        rotation: float = 0.0,
        ellipse: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None,
        circularity: float = 0.0,
        darkness: float = 0.0,
        confidence: float = 0.0,
        quality_score: float = 0.0,
        inlier_ratio: float = 0.0,
        mask: Optional[np.ndarray] = None,
        starburst_inliers: Optional[List[Tuple[float, float]]] = None,
        all_candidates: Optional[List[Dict[str, Any]]] = None,
        detected: bool = False,
        message: str = "Pupil detected successfully.",
        status: str = "NOT_DETECTED",
        confidence_breakdown: Optional[Dict[str, Any]] = None
    ):
        self.center = center
        self.radius = radius
        self.radius_x = radius_x if radius_x > 0 else radius
        self.radius_y = radius_y if radius_y > 0 else radius
        self.rotation = rotation
        self.ellipse = ellipse  # ((cx, cy), (w, h), angle)
        self.circularity = circularity
        self.darkness = darkness
        self.confidence = confidence
        self.quality_score = quality_score
        self.inlier_ratio = inlier_ratio
        self.mask = mask
        self.starburst_inliers = starburst_inliers or []
        self.all_candidates = all_candidates or []
        self.detected = detected
        self.message = message
        self.status = status
        self.confidence_breakdown = confidence_breakdown or {}


def detect_pupil(
    prep_dict: Dict[str, Any],
    eyelid_mask: Optional[np.ndarray] = None,
    glint_mask: Optional[np.ndarray] = None,
    sensitivity: float = 0.5,
    fit_mode: str = "ellipse",
    use_starburst_refinement: bool = True,
    prev_pupil_center: Optional[Tuple[float, float]] = None
) -> PupilResult:
    """
    Detects pupil boundary using a multi-candidate ensemble and RANSAC ellipse fitting.
    """
    gray = prep_dict.get("bilateral_raw", prep_dict.get("gray_inpainted", prep_dict.get("gray")))
    gray_clean = prep_dict.get("gray_inpainted", gray)
    blurred = prep_dict.get("blurred", gray)
    h, w = gray.shape[:2]
    min_dim = float(min(h, w))

    # Physical pupil dimensions relative to frame
    min_radius = max(8.0, min_dim * PUPIL_MIN_RADIUS_RATIO)
    max_radius = min_dim * PUPIL_MAX_RADIUS_RATIO
    min_area = np.pi * (min_radius ** 2) * 0.45
    max_area = np.pi * (max_radius ** 2) * 1.30

    candidates = []

    # ---------------------------------------------------------
    # Generator 1: Multi-Threshold Candidate Sweep over Percentiles
    # [5, 8, 12, 18, 25, 35]
    # ---------------------------------------------------------
    for pct in CANDIDATE_PERCENTILES:
        thresh = float(np.percentile(gray, pct))
        # Account for user sensitivity
        thresh = thresh + (sensitivity - 0.5) * 12.0
        thresh = float(np.clip(thresh, 5.0, 210.0))

        _, binary = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY_INV)

        # Morphological open + close: fill glints and detach thin eyelash threads
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_clean = cv2.morphologyEx(binary_filled, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(binary_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 5:
                continue

            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Fit ellipse to contour
            try:
                ((ecx, ecy), (ew, eh), eang) = cv2.fitEllipse(cnt)
            except Exception:
                continue

            rx = float(ew / 2.0)
            ry = float(eh / 2.0)
            r = float((rx + ry) / 2.0)

            if r < min_radius or r > max_radius:
                continue

            axis_ratio = min(rx, ry) / max(rx, ry)
            if axis_ratio < PUPIL_MIN_AXIS_RATIO:
                continue

            # Deduplication: check if similar candidate already recorded
            is_duplicate = False
            for existing in candidates:
                ec = existing["center"]
                dist = math.hypot(ecx - ec[0], ecy - ec[1])
                r_diff = abs(r - existing["radius"]) / max(1.0, existing["radius"])
                if dist < 5.0 and r_diff < 0.18:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_val = float(np.mean(gray[cnt_mask == 255])) if np.any(cnt_mask == 255) else 128.0

            cand_dict = {
                "contour": cnt,
                "center": (int(round(ecx)), int(round(ecy))),
                "radius": r,
                "rx": rx,
                "ry": ry,
                "angle": eang,
                "area": area,
                "circularity": axis_ratio,
                "darkness": mean_val,
                "generator": f"percentile_{pct}",
                "mask": cnt_mask
            }

            cand_score, breakdown = score_pupil_candidate(
                cand_dict, gray_clean, (h, w), eyelid_mask, glint_mask, prev_pupil_center, return_breakdown=True
            )
            cand_dict["score"] = cand_score
            cand_dict["breakdown"] = breakdown

            # Only retain candidates that passed hard physical plausibility (score > 0.0)
            if cand_score > 0.10:
                candidates.append(cand_dict)

    # ---------------------------------------------------------
    # Generator 2: Circular Hough Transform (Secondary / Fallback)
    # ---------------------------------------------------------
    hough_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w * 0.2,
        param1=50,
        param2=int(32 - 8 * sensitivity),
        minRadius=int(min_radius),
        maxRadius=int(max_radius)
    )

    if hough_circles is not None:
        hough_circles = np.round(hough_circles[0, :]).astype(int)
        for (hx, hy, hr) in hough_circles:
            # Plausibility check on Hough circle
            is_dup = any(math.hypot(hx - c["center"][0], hy - c["center"][1]) < 8.0 for c in candidates)
            if not is_dup:
                cand_dict = {
                    "contour": None,
                    "center": (int(hx), int(hy)),
                    "radius": float(hr),
                    "rx": float(hr),
                    "ry": float(hr),
                    "angle": 0.0,
                    "area": np.pi * hr**2,
                    "circularity": 1.0,
                    "darkness": 128.0,
                    "generator": "hough",
                    "mask": None
                }
                cand_score, breakdown = score_pupil_candidate(
                    cand_dict, gray_clean, (h, w), eyelid_mask, glint_mask, prev_pupil_center, return_breakdown=True
                )
                cand_dict["score"] = cand_score
                cand_dict["breakdown"] = breakdown
                if cand_score > 0.15:
                    candidates.append(cand_dict)

    if not candidates:
        return PupilResult(
            center=(w // 2, h // 2),
            radius=0.0,
            detected=False,
            message="No valid pupil candidates found. Hard plausibility or contrast checks failed.",
            status="NOT_DETECTED",
            confidence=0.0
        )

    # Sort candidates by composite multi-factor score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    initial_center = best["center"]
    initial_radius = best["radius"]
    best_cnt = best["contour"]
    fitted_ellipse = None
    starburst_inliers = []
    rx = best.get("rx", initial_radius)
    ry = best.get("ry", initial_radius)
    rotation = best.get("angle", 0.0)

    # Robust Ellipse Fitting on Best Candidate Contour
    if best_cnt is not None and len(best_cnt) >= 5:
        pts = best_cnt.reshape(-1, 2)
        ransac_fit = ransac_ellipse_fit(pts, max_iterations=150, inlier_threshold=3.0, min_inliers=5)
        if ransac_fit["success"] and ransac_fit["ellipse"] is not None:
            fitted_ellipse = ransac_fit["ellipse"]
            (ecx, ecy), (ew, eh), eang = fitted_ellipse
            initial_center = (int(round(ecx)), int(round(ecy)))
            rx = ew / 2.0
            ry = eh / 2.0
            rotation = eang
            initial_radius = (rx + ry) / 2.0

    # Starburst Sub-Pixel Refinement
    if use_starburst_refinement:
        starburst_res = refine_pupil_boundary_starburst(
            gray_clean,
            initial_center,
            initial_radius,
            num_rays=48,
            ransac_iterations=100
        )
        final_center = starburst_res["refined_center"]
        final_radius = starburst_res["refined_radius"]
        if starburst_res["ellipse"] is not None:
            fitted_ellipse = starburst_res["ellipse"]
            (ecx, ecy), (ew, eh), eang = fitted_ellipse
            rx = ew / 2.0
            ry = eh / 2.0
            rotation = eang
        starburst_inliers = starburst_res["inlier_points"]
    else:
        final_center = initial_center
        final_radius = initial_radius

    if fit_mode == "circle":
        rx = final_radius
        ry = final_radius
        rotation = 0.0
        fitted_ellipse = ((float(final_center[0]), float(final_center[1])), (float(final_radius * 2), float(final_radius * 2)), 0.0)

    # Final Geometric Plausibility Check
    final_pupil_dict = {
        "cx": float(final_center[0]),
        "cy": float(final_center[1]),
        "rx": float(rx),
        "ry": float(ry),
        "radius": float(final_radius),
        "angle": float(rotation),
        "detected": True
    }
    is_valid, failures, plaus_score = validate_detection(final_pupil_dict, iris=None, frame_shape=(h, w))

    # Calibrated Confidence Fusion
    cand_bd = best.get("breakdown", {})
    g_sup = cand_bd.get("gradient", 0.70)
    d_con = cand_bd.get("contrast", 0.70)
    c_comp = cand_bd.get("completeness", 0.70)
    inlier_r = float(len(starburst_inliers) / 48.0) if starburst_inliers else 0.75

    confidence, conf_breakdown = compute_calibrated_confidence(
        gradient_support=g_sup,
        darkness_contrast=d_con,
        contour_completeness=c_comp,
        inlier_ratio=inlier_r,
        plausibility_score=plaus_score,
        is_valid=is_valid
    )
    conf_breakdown["failures"] = failures

    # Generate pupil binary mask
    pupil_mask = np.zeros((h, w), dtype=np.uint8)
    if is_valid:
        if fitted_ellipse is not None:
            try:
                cv2.ellipse(pupil_mask, fitted_ellipse, 255, -1)
            except Exception:
                cv2.circle(pupil_mask, final_center, int(round(final_radius)), 255, -1)
        else:
            cv2.circle(pupil_mask, final_center, int(round(final_radius)), 255, -1)

    detected = is_valid and (confidence >= 35.0)
    status = "DETECTED" if confidence >= 55.0 else ("LOW_CONFIDENCE" if detected else "NOT_DETECTED")

    return PupilResult(
        center=final_center if is_valid else (w // 2, h // 2),
        radius=final_radius if is_valid else 0.0,
        radius_x=rx if is_valid else 0.0,
        radius_y=ry if is_valid else 0.0,
        rotation=rotation if is_valid else 0.0,
        ellipse=fitted_ellipse if is_valid else None,
        circularity=best["circularity"],
        darkness=best["darkness"],
        confidence=confidence,
        quality_score=confidence,
        inlier_ratio=inlier_r,
        mask=pupil_mask,
        starburst_inliers=starburst_inliers,
        all_candidates=candidates[:8],
        detected=detected,
        message="Pupil detected successfully." if detected else f"Pupil validation failed: {', '.join(failures) if failures else 'Low confidence'}",
        status=status,
        confidence_breakdown=conf_breakdown
    )
