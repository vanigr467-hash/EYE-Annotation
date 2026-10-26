"""
Constrained multi-method iris detector.

Searches for the iris-sclera boundary (limbus) ONLY within a constrained
annular region around the detected pupil. Uses multiple candidate generation
methods and scores them to select the best iris boundary.

Key design principles:
1. ANNULAR CONSTRAINT: search only between pupil_radius*1.3 and pupil_radius*3.5
2. EYELID MASKING: exclude eyelid-occluded regions before fitting
3. MULTI-METHOD: radial rays + integro-differential + edge contours
4. TEXTURE VERIFICATION: check that inside/outside boundary look different
5. FAIL-SAFE: prefer NOT_DETECTED over wrong detection
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math
from src.ellipse_fitting import fit_ellipse_fitzgibbon, generate_ellipse_points
from src.config_constants import (
    IRIS_MIN_RADIUS_RATIO,
    IRIS_MAX_RADIUS_RATIO,
    MAX_PUPIL_IRIS_CENTER_OFFSET_RATIO,
    MIN_PUPIL_IRIS_RADIUS_RATIO,
    MAX_PUPIL_IRIS_RADIUS_RATIO
)
from src.plausibility import validate_detection


class IrisResult:
    """Stores iris boundary detection results."""

    def __init__(
        self,
        center: Tuple[int, int] = (0, 0),
        radius: float = 0.0,
        radius_x: float = 0.0,
        radius_y: float = 0.0,
        rotation: float = 0.0,
        ellipse: Optional[Tuple] = None,
        confidence: float = 0.0,
        quality_score: float = 0.0,
        edge_support: float = 0.0,
        inlier_ratio: float = 0.0,
        num_valid_rays: int = 0,
        boundary_candidates: Optional[List[Tuple[float, float]]] = None,
        inlier_points: Optional[List[Tuple[float, float]]] = None,
        outlier_points: Optional[List[Tuple[float, float]]] = None,
        mask: Optional[np.ndarray] = None,
        detection_method: str = "",
        detected: bool = False,
        message: str = "",
        status: str = "NOT_DETECTED"
    ):
        self.center = center
        self.radius = radius
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.rotation = rotation
        self.ellipse = ellipse
        self.confidence = confidence
        self.quality_score = quality_score
        self.edge_support = edge_support
        self.inlier_ratio = inlier_ratio
        self.num_valid_rays = num_valid_rays
        self.boundary_candidates = boundary_candidates or []
        self.inlier_points = inlier_points or []
        self.outlier_points = outlier_points or []
        self.mask = mask
        self.detection_method = detection_method
        self.detected = detected
        self.message = message
        self.status = status  # DETECTED / LOW_CONFIDENCE / PARTIAL / NOT_DETECTED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUM_RAYS = 120
_MIN_RATIO = 1.4      # minimum iris_radius / pupil_radius
_MAX_RATIO = 4.0       # maximum iris_radius / pupil_radius
_RANSAC_ITERS = 250
_RANSAC_THRESH = 4.0
_EYELID_HALF_DEG = 40.0  # exclude ±40° around top/bottom for eyelids


def detect_iris(
    prep_dict: Dict[str, Any],
    pupil_input: Any,
    pupil_radius: Optional[float] = None,
    eyelid_mask: Optional[np.ndarray] = None,
    sensitivity: float = 0.5,
    fit_mode: str = "ellipse",
    prev_iris: Optional[Dict] = None
) -> IrisResult:
    """
    Detects the iris boundary using a constrained multi-method approach.
    """
    gray = prep_dict.get("bilateral_raw", prep_dict.get("gray_inpainted", prep_dict.get("gray")))
    clahe = prep_dict.get("clahe", gray)
    h, w = gray.shape[:2]

    # Priority 4: If pupil is not detected or invalid, emit null iris with "no valid pupil seed"
    if pupil_input is None:
        return _fail("no valid pupil seed")

    if hasattr(pupil_input, "detected"):
        if not pupil_input.detected or pupil_input.radius <= 0:
            return _fail("no valid pupil seed")
        px, py = int(pupil_input.center[0]), int(pupil_input.center[1])
        pr = float(max(3.0, pupil_input.radius))
    else:
        px, py = int(pupil_input[0]), int(pupil_input[1])
        pr = float(max(3.0, pupil_radius if pupil_radius is not None else 10.0))

    p_valid, p_fails, _ = validate_detection(pupil_input, iris=None, frame_shape=(h, w))
    if not p_valid:
        return _fail("no valid pupil seed")

    # Annular search bounds (Strict Pupil-Dependent Constraint)
    r_inner = pr * _MIN_RATIO
    r_outer = min(pr * _MAX_RATIO, min(w, h) * IRIS_MAX_RADIUS_RATIO)

    if r_inner >= r_outer:
        return _fail("no valid pupil seed")

    # If we have previous iris, use it to constrain the search
    if prev_iris is not None and prev_iris.get("radius", 0) > 0:
        prev_r = prev_iris["radius"]
        # Allow ±25% variation from previous frame
        r_inner = max(r_inner, prev_r * 0.75)
        r_outer = min(r_outer, prev_r * 1.25)

    # Compute gradient images
    blurred_1 = cv2.GaussianBlur(clahe, (5, 5), 1.5)
    blurred_2 = cv2.GaussianBlur(clahe, (9, 9), 3.0)
    grad_x = cv2.Sobel(blurred_1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_1, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag_smooth = cv2.GaussianBlur(grad_mag, (5, 5), 1.5)

    # --- METHOD A: Radial ray gradient search ---
    method_a_result = _radial_ray_search(
        gray, clahe, grad_x, grad_y, grad_mag_smooth,
        px, py, pr, r_inner, r_outer, eyelid_mask, h, w, sensitivity
    )

    # --- METHOD B: Integro-differential operator ---
    method_b_result = _integro_differential_search(
        blurred_2, px, py, pr, r_inner, r_outer, h, w
    )

    # --- METHOD C: Edge contour search ---
    method_c_result = _edge_contour_search(
        clahe, px, py, pr, r_inner, r_outer, eyelid_mask, h, w
    )

    # --- Score and select best candidate ---
    candidates = []
    for method_name, result in [("radial_rays", method_a_result),
                                  ("integro_diff", method_b_result),
                                  ("edge_contour", method_c_result)]:
        if result is not None:
            result["method"] = method_name
            # Texture verification score
            texture_score = _verify_iris_sclera_texture(
                gray, px, py, result["radius"], result.get("ellipse")
            )
            result["texture_score"] = texture_score
            result["total_score"] = (
                result.get("edge_score", 0) * 0.35 +
                result.get("inlier_score", 0) * 0.25 +
                texture_score * 0.25 +
                result.get("geometry_score", 0) * 0.15
            )
            candidates.append(result)

    if not candidates:
        return _fail("No iris boundary candidates found.")

    # Select best candidate
    candidates.sort(key=lambda c: c["total_score"], reverse=True)
    best = candidates[0]

    if best["total_score"] < 0.15:
        return _fail("All iris candidates scored too low. Detection unreliable.",
                      boundary_candidates=best.get("all_points", []),
                      outlier_points=best.get("outlier_points", []))

    # Extract final ellipse
    ell = best.get("ellipse")
    if ell is None:
        return _fail("Ellipse fitting failed for best candidate.")

    (ecx, ecy), (ew, eh), eang = ell
    erx, ery = ew / 2.0, eh / 2.0
    mean_r = (erx + ery) / 2.0

    if fit_mode == "circle":
        erx = ery = mean_r
        ew = eh = mean_r * 2.0
        eang = 0.0
        ell = ((ecx, ecy), (ew, eh), eang)

    # Final validation via physical plausibility constraints
    final_iris_dict = {
        "cx": float(ecx), "cy": float(ecy),
        "rx": float(erx), "ry": float(ery),
        "radius": float(mean_r), "angle": float(eang),
        "detected": True
    }
    is_valid, failures, plaus_score = validate_detection(pupil_input, final_iris_dict, (h, w))
    if not is_valid:
        return _fail(f"Geometric constraint violation: {', '.join(failures)}",
                     boundary_candidates=best.get("all_points", []),
                     outlier_points=best.get("outlier_points", []))

    # Compute quality and confidence
    quality = round(best["total_score"] * 100.0, 1)
    inlier_ratio = best.get("inlier_ratio", 0.0)
    edge_support = best.get("edge_score", 0.0)

    # Confidence: strictly gated by plausibility
    confidence = float(np.clip(quality * plaus_score, 10.0, 95.0))

    # Determine status
    if confidence >= 60.0:
        status = "DETECTED"
    elif confidence >= 35.0:
        status = "LOW_CONFIDENCE"
    else:
        status = "NOT_DETECTED"

    detected = status in ("DETECTED", "LOW_CONFIDENCE")

    # Build mask
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    if ell is not None:
        try:
            cv2.ellipse(iris_mask, ell, 255, -1)
        except Exception:
            pass

    icx, icy = int(round(ecx)), int(round(ecy))
    inliers = best.get("inlier_points", [])
    if len(inliers) < 10 and detected:
        ell_pts = generate_ellipse_points(ecx, ecy, erx, ery, eang, num_points=64)
        inliers = [(float(p[0]), float(p[1])) for p in ell_pts if (0 <= p[0] < w and 0 <= p[1] < h)]

    return IrisResult(
        center=(icx, icy),
        radius=mean_r,
        radius_x=float(erx),
        radius_y=float(ery),
        rotation=float(eang),
        ellipse=ell,
        confidence=confidence,
        quality_score=quality,
        edge_support=round(edge_support * 100, 1),
        inlier_ratio=round(inlier_ratio, 3),
        num_valid_rays=best.get("num_valid", 0),
        boundary_candidates=best.get("all_points", []),
        inlier_points=inliers,
        outlier_points=best.get("outlier_points", []),
        mask=iris_mask,
        detection_method=best["method"],
        detected=detected,
        message="Iris boundary detected." if detected else f"Low confidence: {best.get('reason', '')}",
        status=status
    )


# ---------------------------------------------------------------------------
# Method A: Radial Ray Gradient Search
# ---------------------------------------------------------------------------

def _radial_ray_search(
    gray, clahe, grad_x, grad_y, grad_mag_smooth,
    px, py, pr, r_inner, r_outer, eyelid_mask, h, w, sensitivity
) -> Optional[Dict]:
    """Search along radial rays for the iris-sclera boundary."""
    angles = np.linspace(0, 2 * np.pi, _NUM_RAYS, endpoint=False)
    boundary_pts = []
    rejected_pts = []
    all_pts = []

    for angle in angles:
        angle_deg = np.degrees(angle) % 360.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Skip eyelid zones
        if _is_eyelid_angle(angle_deg):
            continue

        # Sample along ray from r_inner to r_outer
        num_samples = max(25, int((r_outer - r_inner) * 0.7))
        radii = np.linspace(r_inner, r_outer, num_samples)

        # Find the BEST iris-sclera gradient peak
        # Look for the strongest dark-to-light transition within the annulus
        best_grad = 0.0
        best_r = -1.0
        best_pt = None

        # Collect gradient profile
        ray_profile = []
        for r in radii:
            sx = px + r * cos_a
            sy = py + r * sin_a
            xi, yi = int(round(sx)), int(round(sy))

            if xi < 1 or xi >= w - 1 or yi < 1 or yi >= h - 1:
                break

            # Check eyelid mask
            if eyelid_mask is not None and eyelid_mask[yi, xi] == 0:
                continue

            g = grad_mag_smooth[yi, xi]
            # Directional: must be outward dark-to-light
            rg = grad_x[yi, xi] * cos_a + grad_y[yi, xi] * sin_a

            ray_profile.append((r, sx, sy, g, rg))

        if len(ray_profile) < 5:
            continue

        # Find peaks in the gradient profile
        grads = np.array([p[3] for p in ray_profile])
        radial_grads = np.array([p[4] for p in ray_profile])

        for k in range(1, len(grads) - 1):
            if (grads[k] > grads[k-1] and grads[k] >= grads[k+1] and
                radial_grads[k] > 0 and grads[k] > 3.0):
                r, sx, sy = ray_profile[k][0], ray_profile[k][1], ray_profile[k][2]

                # Score this peak: prefer moderate radius (typical iris-pupil ratio ~2-3x)
                ratio = r / pr if pr > 0 else 0
                ratio_bonus = 1.0 if (1.8 <= ratio <= 3.2) else 0.5
                score = grads[k] * ratio_bonus

                if score > best_grad:
                    best_grad = score
                    best_r = r
                    best_pt = (sx, sy)

        if best_pt is not None:
            all_pts.append(best_pt)

            # Check if within reasonable bounds
            if r_inner <= best_r <= r_outer:
                boundary_pts.append(best_pt)
            else:
                rejected_pts.append(best_pt)

    if len(boundary_pts) < 8:
        return None

    pts_arr = np.array(boundary_pts, dtype=np.float32)

    # RANSAC ellipse fitting
    fit_result = _ransac_fit(pts_arr, px, py, pr, r_outer)
    if fit_result is None:
        return None

    inlier_pts = [(float(p[0]), float(p[1])) for p in pts_arr[fit_result["inlier_mask"]]]
    outlier_pts = rejected_pts + [(float(p[0]), float(p[1]))
                                   for p in pts_arr[~fit_result["inlier_mask"]]]

    return {
        "ellipse": fit_result["ellipse"],
        "radius": fit_result["mean_radius"],
        "inlier_ratio": fit_result["inlier_ratio"],
        "inlier_score": fit_result["inlier_ratio"],
        "edge_score": min(1.0, len(boundary_pts) / 50.0),
        "geometry_score": fit_result.get("geometry_score", 0.5),
        "num_valid": len(boundary_pts),
        "all_points": [(float(p[0]), float(p[1])) for p in all_pts],
        "inlier_points": inlier_pts,
        "outlier_points": outlier_pts
    }


# ---------------------------------------------------------------------------
# Method B: Integro-Differential Operator (Daugman-style)
# ---------------------------------------------------------------------------

def _integro_differential_search(
    blurred, px, py, pr, r_inner, r_outer, h, w
) -> Optional[Dict]:
    """
    Daugman-style integro-differential operator:
    Find the radius that maximizes |d/dr integral(I on circle(r))| / (2*pi*r)
    """
    best_contrast = 0.0
    best_r = 0.0

    # Test radii in the annular range
    step = max(1, int((r_outer - r_inner) / 40))
    test_radii = np.arange(r_inner, r_outer, step)

    if len(test_radii) < 3:
        return None

    # Compute mean intensity on circles of different radii
    prev_mean = None
    contrast_scores = []

    for r in test_radii:
        # Sample points on the circle (only lateral sectors, avoid eyelids)
        angles = []
        for a_deg in range(0, 360, 5):
            if not _is_eyelid_angle(float(a_deg)):
                angles.append(np.radians(a_deg))

        if len(angles) < 10:
            continue

        intensities = []
        for a in angles:
            sx = int(round(px + r * np.cos(a)))
            sy = int(round(py + r * np.sin(a)))
            if 0 <= sx < w and 0 <= sy < h:
                intensities.append(float(blurred[sy, sx]))

        if len(intensities) < 5:
            continue

        curr_mean = float(np.mean(intensities))

        if prev_mean is not None:
            # The iris-sclera boundary is where intensity changes most rapidly
            contrast = abs(curr_mean - prev_mean) / max(1.0, step)
            contrast_scores.append((r, contrast, curr_mean))

            if contrast > best_contrast:
                best_contrast = contrast
                best_r = r

        prev_mean = curr_mean

    if best_r <= 0 or best_contrast < 0.5:
        return None

    # Create a circle candidate at the best radius
    ell = ((float(px), float(py)), (float(best_r * 2), float(best_r * 2)), 0.0)

    # Geometry score
    ratio = best_r / pr if pr > 0 else 0
    geo_score = 1.0 if (1.8 <= ratio <= 3.2) else (0.5 if (1.3 <= ratio <= 4.0) else 0.2)

    return {
        "ellipse": ell,
        "radius": best_r,
        "inlier_ratio": 0.6,  # estimated
        "inlier_score": min(1.0, best_contrast / 3.0),
        "edge_score": min(1.0, best_contrast / 3.0),
        "geometry_score": geo_score,
        "num_valid": len(contrast_scores),
        "all_points": [],
        "inlier_points": [],
        "outlier_points": []
    }


# ---------------------------------------------------------------------------
# Method C: Edge Contour Search
# ---------------------------------------------------------------------------

def _edge_contour_search(
    clahe, px, py, pr, r_inner, r_outer, eyelid_mask, h, w
) -> Optional[Dict]:
    """Search for iris boundary by finding elliptical contours in the annular region."""

    # Create annular mask
    annular_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(annular_mask, (px, py), int(r_outer), 255, -1)
    cv2.circle(annular_mask, (px, py), int(r_inner), 0, -1)

    # Apply eyelid mask
    if eyelid_mask is not None:
        annular_mask = cv2.bitwise_and(annular_mask, eyelid_mask)

    # Edge detection within the annular region
    blurred = cv2.GaussianBlur(clahe, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 80)
    edges = cv2.bitwise_and(edges, annular_mask)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Collect all edge points in the annular region
    all_edge_pts = []
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if r_inner <= dist <= r_outer:
                all_edge_pts.append((float(x), float(y)))

    if len(all_edge_pts) < 15:
        return None

    pts_arr = np.array(all_edge_pts, dtype=np.float32)

    # RANSAC ellipse fitting
    fit_result = _ransac_fit(pts_arr, px, py, pr, r_outer)
    if fit_result is None:
        return None

    inlier_pts = [(float(p[0]), float(p[1])) for p in pts_arr[fit_result["inlier_mask"]]]

    return {
        "ellipse": fit_result["ellipse"],
        "radius": fit_result["mean_radius"],
        "inlier_ratio": fit_result["inlier_ratio"],
        "inlier_score": fit_result["inlier_ratio"],
        "edge_score": min(1.0, len(all_edge_pts) / 80.0),
        "geometry_score": fit_result.get("geometry_score", 0.5),
        "num_valid": len(all_edge_pts),
        "all_points": all_edge_pts[:200],  # limit for display
        "inlier_points": inlier_pts,
        "outlier_points": []
    }


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _ransac_fit(
    pts_arr: np.ndarray,
    px: int, py: int, pr: float, r_outer: float
) -> Optional[Dict]:
    """RANSAC ellipse fitting with geometric validation."""
    n = len(pts_arr)
    if n < 5:
        return None

    best_ellipse = None
    best_inlier_mask = None
    best_inlier_count = 0

    for _ in range(_RANSAC_ITERS):
        idx = np.random.choice(n, size=min(6, n), replace=False)
        sample = pts_arr[idx]

        ell = fit_ellipse_fitzgibbon(sample)
        if ell is None:
            continue

        (ecx, ecy), (ew, eh), eang = ell
        erx, ery = ew / 2.0, eh / 2.0

        # Quick validation
        if erx < pr * 0.8 or ery < pr * 0.8:
            continue
        if erx > r_outer * 1.3 or ery > r_outer * 1.3:
            continue
        if max(erx, ery) / max(1.0, min(erx, ery)) > 2.0:
            continue
        cdist = np.sqrt((ecx - px) ** 2 + (ecy - py) ** 2)
        if cdist > pr * MAX_PUPIL_IRIS_CENTER_OFFSET_RATIO:
            continue

        # Compute distances
        dists = _point_to_ellipse_dist(pts_arr, ecx, ecy, erx, ery, eang)
        inlier_mask = dists < _RANSAC_THRESH
        count = int(np.sum(inlier_mask))

        if count > best_inlier_count:
            best_inlier_count = count
            best_inlier_mask = inlier_mask
            best_ellipse = ell

    # Refit on inliers with Fitzgibbon
    if best_ellipse is not None and best_inlier_mask is not None and best_inlier_count >= 5:
        inlier_pts = pts_arr[best_inlier_mask]
        refit_ell = fit_ellipse_fitzgibbon(inlier_pts)
        if refit_ell is not None:
            best_ellipse = refit_ell

    if best_ellipse is None:
        # Fallback: fit on all points
        if n >= 5:
            best_ellipse = fit_ellipse_fitzgibbon(pts_arr)
            if best_ellipse is not None:
                best_inlier_mask = np.ones(n, dtype=bool)
                best_inlier_count = n
            else:
                return None
        else:
            return None

    (ecx, ecy), (ew, eh), eang = best_ellipse
    erx, ery = ew / 2.0, eh / 2.0
    mean_r = (erx + ery) / 2.0

    ratio = mean_r / pr if pr > 0 else 0
    geo_score = 1.0 if (1.8 <= ratio <= 3.2) else (0.5 if (1.3 <= ratio <= 4.0) else 0.2)

    return {
        "ellipse": best_ellipse,
        "mean_radius": mean_r,
        "inlier_mask": best_inlier_mask,
        "inlier_count": best_inlier_count,
        "inlier_ratio": best_inlier_count / float(max(1, n)),
        "geometry_score": geo_score
    }


def _point_to_ellipse_dist(points, cx, cy, rx, ry, angle_deg):
    """Approximate point-to-ellipse distance using polar comparison."""
    if rx <= 0 or ry <= 0:
        return np.full(len(points), 999.0)

    a_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy

    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a

    theta = np.arctan2(ly, lx)
    denom = np.sqrt((ry * np.cos(theta)) ** 2 + (rx * np.sin(theta)) ** 2)
    denom = np.maximum(denom, 1e-6)
    ellipse_r = (rx * ry) / denom
    point_r = np.sqrt(lx ** 2 + ly ** 2)

    return np.abs(point_r - ellipse_r)


def _verify_iris_sclera_texture(
    gray: np.ndarray,
    px: int, py: int,
    radius: float,
    ellipse: Optional[Tuple] = None
) -> float:
    """
    Verifies that inside the boundary looks like iris and outside looks like sclera.
    Supports off-axis gazes by checking both lateral sectors (nasal and temporal).
    Returns 0-1 score (1 = strong iris-sclera contrast).
    """
    h, w = gray.shape[:2]
    r = radius
    if r <= 5:
        return 0.0

    left_inner, left_outer = [], []
    right_inner, right_outer = [], []

    for a_deg in range(0, 360, 10):
        if _is_eyelid_angle(float(a_deg)):
            continue
        a_rad = np.radians(a_deg)
        cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)

        ix = int(round(px + r * 0.82 * cos_a))
        iy = int(round(py + r * 0.82 * sin_a))
        ox = int(round(px + r * 1.18 * cos_a))
        oy = int(round(py + r * 1.18 * sin_a))

        if 0 <= ix < w and 0 <= iy < h and 0 <= ox < w and 0 <= oy < h:
            val_in = float(gray[iy, ix])
            val_out = float(gray[oy, ox])
            if cos_a > 0:  # Right sector
                right_inner.append(val_in)
                right_outer.append(val_out)
            else:          # Left sector
                left_inner.append(val_in)
                left_outer.append(val_out)

    contrasts = []
    if len(left_inner) >= 3 and len(left_outer) >= 3:
        contrasts.append(float(np.mean(left_outer)) - float(np.mean(left_inner)))
    if len(right_inner) >= 3 and len(right_outer) >= 3:
        contrasts.append(float(np.mean(right_outer)) - float(np.mean(right_inner)))

    if not contrasts:
        return 0.4  # neutral score when lateral evidence is limited

    best_contrast = max(contrasts)
    if best_contrast <= 0:
        return 0.1  # outer is darker than inner

    score = min(1.0, max(0.0, best_contrast / 30.0))
    return score


def _is_eyelid_angle(angle_deg: float) -> bool:
    """
    Restricts search to left/right lateral horizontal sectors (±30° around horizontal).
    Angles in [330°, 360°], [0°, 30°] (right sector) and [150°, 210°] (left sector) are valid (return False).
    All other angles (upper and lower eyelid zones) are excluded (return True).
    """
    a = float(angle_deg) % 360.0
    if a <= 30.0 or a >= 330.0:
        return False
    if 150.0 <= a <= 210.0:
        return False
    return True


def _fail(
    message: str,
    boundary_candidates=None,
    outlier_points=None
) -> IrisResult:
    """Create a failed detection result."""
    return IrisResult(
        detected=False,
        message=message,
        status="NOT_DETECTED",
        boundary_candidates=boundary_candidates or [],
        outlier_points=outlier_points or []
    )
