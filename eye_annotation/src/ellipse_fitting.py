"""
Shared robust ellipse fitting and boundary point refinement utility
used by both pupil and iris detectors.
"""

from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


def point_to_ellipse_distance(
    points: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float
) -> np.ndarray:
    """
    Computes approximate Euclidean distance from each point to the ellipse boundary.
    Uses polar-radius comparison in the ellipse's local rotated coordinate frame.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        cx, cy: ellipse center
        rx, ry: semi-major / semi-minor axis lengths (half of width/height)
        angle_deg: ellipse rotation angle in degrees
        
    Returns:
        (N,) array of distances
    """
    if len(points) == 0:
        return np.array([], dtype=np.float64)
    if rx <= 0 or ry <= 0:
        return np.full(len(points), 999.0)

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy

    # Rotate into ellipse-aligned local frame
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a

    theta = np.arctan2(local_y, local_x)
    denom = np.sqrt((ry * np.cos(theta)) ** 2 + (rx * np.sin(theta)) ** 2)
    denom = np.maximum(denom, 1e-7)
    ellipse_r = (rx * ry) / denom
    point_r = np.sqrt(local_x ** 2 + local_y ** 2)

    return np.abs(point_r - ellipse_r)


def fit_ellipse_fitzgibbon(points: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Direct Least Squares Fitting of Ellipses (Fitzgibbon, Pilu, Fisher 1999).
    Guaranteed to return an exact ellipse (satisfying 4AC - B^2 = 1) even on partial arcs.
    Uses OpenCV's cv2.fitEllipseDirect with fallback to cv2.fitEllipse.
    
    Args:
        points: (N, 2) array of (x, y) coordinates (N >= 5)
        
    Returns:
        ((cx, cy), (width, height), angle_deg) in OpenCV ellipse format, or None
    """
    if points is None or len(points) < 5:
        return None

    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        pts = pts.reshape(-1, 2)

    try:
        return cv2.fitEllipseDirect(pts)
    except Exception:
        try:
            return cv2.fitEllipse(pts)
        except Exception:
            return None


def ransac_ellipse_fit(
    points: np.ndarray,
    max_iterations: int = 200,
    inlier_threshold: float = 3.0,
    min_inliers: int = 6
) -> Dict[str, Any]:
    """
    Fits an ellipse to 2D boundary points using RANSAC outlier pruning with Fitzgibbon fitting.
    """
    failure_result = {
        "ellipse": None,
        "inlier_mask": np.zeros(len(points) if points is not None else 0, dtype=bool),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "residual": 999.0,
        "success": False
    }

    if points is None or len(points) < 5:
        return failure_result

    pts_arr = np.asarray(points, dtype=np.float32)
    n_pts = len(pts_arr)

    # Point cloud bounding box for center validation
    min_x, min_y = np.min(pts_arr, axis=0) - 20
    max_x, max_y = np.max(pts_arr, axis=0) + 20

    best_ellipse = None
    best_inlier_mask = np.zeros(n_pts, dtype=bool)
    best_inlier_count = 0

    for _ in range(max_iterations):
        sample_idx = np.random.choice(n_pts, size=min(6, n_pts), replace=False)
        sample_pts = pts_arr[sample_idx]

        ell = fit_ellipse_fitzgibbon(sample_pts)
        if ell is None:
            continue

        (ecx, ecy), (ew, eh), eangle = ell
        erx, ery = ew / 2.0, eh / 2.0

        # Reject impossible geometry
        if erx < 3.0 or ery < 3.0:
            continue
        major = max(erx, ery)
        minor = max(0.1, min(erx, ery))
        if (major / minor) > 3.5:
            continue
        if not (min_x <= ecx <= max_x and min_y <= ecy <= max_y):
            continue

        dists = point_to_ellipse_distance(pts_arr, ecx, ecy, erx, ery, eangle)
        inliers = dists < inlier_threshold
        count = int(np.sum(inliers))

        if count > best_inlier_count:
            best_inlier_count = count
            best_inlier_mask = inliers
            best_ellipse = ell

    # Refit ellipse on all inliers with Fitzgibbon
    if best_ellipse is not None and best_inlier_count >= min_inliers:
        inlier_pts = pts_arr[best_inlier_mask]
        refit_ell = fit_ellipse_fitzgibbon(inlier_pts)
        if refit_ell is not None:
            best_ellipse = refit_ell
            (ecx, ecy), (ew, eh), eangle = best_ellipse
            dists = point_to_ellipse_distance(pts_arr, ecx, ecy, ew / 2.0, eh / 2.0, eangle)
            best_inlier_mask = dists < inlier_threshold
            best_inlier_count = int(np.sum(best_inlier_mask))

    # Fallback to direct least-squares if RANSAC didn't find enough
    if best_ellipse is None and n_pts >= 5:
        try:
            best_ellipse = cv2.fitEllipse(pts_arr)
            (ecx, ecy), (ew, eh), eangle = best_ellipse
            dists = point_to_ellipse_distance(pts_arr, ecx, ecy, ew / 2.0, eh / 2.0, eangle)
            best_inlier_mask = dists < inlier_threshold
            best_inlier_count = int(np.sum(best_inlier_mask))
        except Exception:
            return failure_result

    if best_ellipse is None or best_inlier_count < min_inliers:
        return failure_result

    (ecx, ecy), (ew, eh), eangle = best_ellipse
    inlier_dists = point_to_ellipse_distance(
        pts_arr[best_inlier_mask], ecx, ecy, ew / 2.0, eh / 2.0, eangle
    )
    median_residual = float(np.median(inlier_dists)) if len(inlier_dists) > 0 else 0.0

    return {
        "ellipse": best_ellipse,
        "inlier_mask": best_inlier_mask,
        "inlier_count": best_inlier_count,
        "inlier_ratio": float(best_inlier_count / max(1, n_pts)),
        "residual": round(median_residual, 2),
        "success": True
    }


def validate_ellipse(
    ellipse: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]],
    constraints: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Validates an ellipse against domain-specific geometric constraints.
    
    Args:
        ellipse: ((cx, cy), (w, h), angle_deg)
        constraints: dict with optional keys:
            - min_radius: float
            - max_radius: float
            - max_eccentricity: float (0.0 to 1.0)
            - max_center_offset: float
            - reference_center: Tuple[float, float]
            - image_shape: Tuple[int, int] (h, w)
            
    Returns:
        (is_valid: bool, reason_if_invalid: str)
    """
    if ellipse is None:
        return False, "Ellipse is None."

    try:
        (cx, cy), (w, h), angle = ellipse
    except Exception:
        return False, "Invalid ellipse structure."

    if np.isnan(cx) or np.isnan(cy) or np.isnan(w) or np.isnan(h):
        return False, "Ellipse contains NaN values."

    rx, ry = w / 2.0, h / 2.0
    if rx <= 0 or ry <= 0:
        return False, "Ellipse axes must be strictly positive."

    mean_r = (rx + ry) / 2.0
    minor = min(rx, ry)
    major = max(rx, ry)

    # Radius constraints
    min_r = constraints.get("min_radius")
    if min_r is not None and minor < min_r:
        return False, f"Semi-minor axis ({minor:.1f}px) below minimum ({min_r:.1f}px)."

    max_r = constraints.get("max_radius")
    if max_r is not None and major > max_r:
        return False, f"Semi-major axis ({major:.1f}px) exceeds maximum ({max_r:.1f}px)."

    # Eccentricity constraint: e = sqrt(1 - (b/a)^2)
    max_ecc = constraints.get("max_eccentricity")
    if max_ecc is not None and major > 0:
        ecc = np.sqrt(max(0.0, 1.0 - (minor / major) ** 2))
        if ecc > max_ecc:
            return False, f"Eccentricity ({ecc:.2f}) exceeds maximum allowed ({max_ecc:.2f})."

    # Reference center offset constraint
    ref_center = constraints.get("reference_center")
    max_offset = constraints.get("max_center_offset")
    if ref_center is not None and max_offset is not None:
        dist = np.sqrt((cx - ref_center[0]) ** 2 + (cy - ref_center[1]) ** 2)
        if dist > max_offset:
            return False, f"Center offset ({dist:.1f}px) exceeds threshold ({max_offset:.1f}px)."

    # Image boundary constraints
    img_shape = constraints.get("image_shape")
    if img_shape is not None:
        img_h, img_w = img_shape[:2]
        if cx < 0 or cx >= img_w or cy < 0 or cy >= img_h:
            return False, "Ellipse center lies outside image bounds."

    return True, "Ellipse is valid."


def generate_ellipse_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    num_points: int = 64
) -> List[Tuple[int, int]]:
    """
    Generates N equally spaced boundary points on an ellipse.
    """
    if num_points <= 0 or rx <= 0 or ry <= 0:
        return []

    angle_rad = np.radians(angle_deg)
    cos_rot = np.cos(angle_rad)
    sin_rot = np.sin(angle_rad)

    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = []
    for theta in thetas:
        ex = rx * np.cos(theta)
        ey = ry * np.sin(theta)
        px = cx + ex * cos_rot - ey * sin_rot
        py = cy + ex * sin_rot + ey * cos_rot
        pts.append((int(round(px)), int(round(py))))

    return pts


def refine_points_to_edge(
    points: List[Tuple[int, int]],
    gradient_image: np.ndarray,
    search_radius: int = 5,
    ellipse_center: Optional[Tuple[float, float]] = None
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Refines boundary points by searching locally along the normal direction for 
    the maximum edge gradient response.
    
    Args:
        points: list of initial (x, y) integer boundary points
        gradient_image: 2D float or uint8 gradient magnitude array
        search_radius: pixel search window along normal (+-search_radius)
        ellipse_center: optional (cx, cy) to determine radial normal
        
    Returns:
        (refined_points, edge_strengths)
    """
    if not points:
        return [], []

    h, w = gradient_image.shape[:2]

    if ellipse_center is None:
        cx = float(np.mean([p[0] for p in points]))
        cy = float(np.mean([p[1] for p in points]))
    else:
        cx, cy = ellipse_center

    refined = []
    strengths = []

    for px, py in points:
        # Radial vector from center to point
        vx = px - cx
        vy = py - cy
        length = np.sqrt(vx ** 2 + vy ** 2)

        if length < 1e-4:
            refined.append((px, py))
            strengths.append(0.0)
            continue

        nx = vx / length
        ny = vy / length

        best_x, best_y = px, py
        max_grad = 0.0

        for offset in range(-search_radius, search_radius + 1):
            sx = int(round(px + offset * nx))
            sy = int(round(py + offset * ny))

            if 0 <= sx < w and 0 <= sy < h:
                g = float(gradient_image[sy, sx])
                if g > max_grad:
                    max_grad = g
                    best_x, best_y = sx, sy

        refined.append((best_x, best_y))
        strengths.append(round(max_grad, 2))

    return refined, strengths
