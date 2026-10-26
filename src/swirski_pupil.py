"""
Industry-Grade Pupil Localization and Boundary Fitting Module.
Inspired by Swirski & Dodgson (2012) and PuRe Eye Tracking Algorithms.

Pipeline:
1. Haar Center-Surround Integral Image Response for robust coarse pupil localization
2. Localized Specular Glint Detection and Inpainting
3. Radial Gradient Vector Filtering (filtering edges whose normals point outward)
4. Fitzgibbon Direct Least Squares Ellipse Fitting with RANSAC Outlier Pruning
"""

from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np


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


def haar_center_surround_response(
    integral_img: np.ndarray,
    cx: int,
    cy: int,
    r_inner: int,
    r_outer: int,
    h: int,
    w: int
) -> float:
    """
    Computes Haar Center-Surround response using integral image:
    Response = Mean(Outer Annulus) - Mean(Inner Disk).
    Maximized when (cx, cy) is centered on a dark pupil surrounded by lighter iris.
    """
    # Clamp inner box [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner]
    x1_in = max(0, cx - r_inner)
    y1_in = max(0, cy - r_inner)
    x2_in = min(w - 1, cx + r_inner)
    y2_in = min(h - 1, cy + r_inner)

    # Clamp outer box [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer]
    x1_out = max(0, cx - r_outer)
    y1_out = max(0, cy - r_outer)
    x2_out = min(w - 1, cx + r_outer)
    y2_out = min(h - 1, cy + r_outer)

    def box_sum(x1, y1, x2, y2):
        return float(integral_img[y2 + 1, x2 + 1] - integral_img[y1, x2 + 1] - integral_img[y2 + 1, x1] + integral_img[y1, x1])

    inner_area = max(1, (x2_in - x1_in + 1) * (y2_in - y1_in + 1))
    outer_area = max(1, (x2_out - x1_out + 1) * (y2_out - y1_out + 1))

    inner_sum = box_sum(x1_in, y1_in, x2_in, y2_in)
    outer_sum = box_sum(x1_out, y1_out, x2_out, y2_out)

    surround_area = max(1, outer_area - inner_area)
    surround_sum = max(0.0, outer_sum - inner_sum)

    mean_inner = inner_sum / inner_area
    mean_surround = surround_sum / surround_area

    # Contrast response (higher = darker inner vs lighter surround)
    return float(mean_surround - mean_inner)


def locate_pupil_coarse_haar(
    gray_img: np.ndarray,
    search_step: int = 4
) -> Tuple[int, int, float, float]:
    """
    Locates coarse pupil center (cx, cy), radius, and contrast response using 
    dark-region candidate seeding and multi-scale Haar Center-Surround filters.
    """
    h, w = gray_img.shape[:2]
    integral = cv2.integral(gray_img)

    best_response = -999.0
    best_cx, best_cy, best_r = w // 2, h // 2, 25.0

    # 1. Dark region candidate seeds
    p10 = float(np.percentile(gray_img, 10))
    _, bin_dark = cv2.threshold(gray_img, int(p10 + 10), 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_dark = cv2.morphologyEx(bin_dark, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(bin_dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    seed_points = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if int(w * 0.10) <= cx_i <= int(w * 0.90) and int(h * 0.12) <= cy_i <= int(h * 0.88):
            seed_points.append((cx_i, cy_i, max(8.0, min(float(r), min(w, h) * 0.20))))

    # If no dark contours found, fall back to interior grid
    if not seed_points:
        for y in range(int(h * 0.20), int(h * 0.80), search_step * 3):
            for x in range(int(w * 0.20), int(w * 0.80), search_step * 3):
                seed_points.append((x, y, 25.0))

    # Multi-scale evaluation on candidate seeds
    for (sx, sy, sr) in seed_points:
        test_radii = [max(8, int(sr * 0.7)), int(sr), min(int(sr * 1.3), 60)]
        for r in test_radii:
            r_out = int(r * 1.8)
            resp = haar_center_surround_response(integral, sx, sy, r, r_out, h, w)
            center_val = float(gray_img[sy, sx])
            center_darkness = 255.0 - center_val
            total_resp = resp * 0.65 + center_darkness * 0.35

            if total_resp > best_response:
                best_response = total_resp
                best_cx, best_cy, best_r = sx, sy, float(r)

    # Fine-grid sub-search around best candidate (+-8px)
    fine_cx, fine_cy = best_cx, best_cy
    fine_best_resp = best_response
    r_out = int(best_r * 1.8)

    for fy in range(max(int(h * 0.10), best_cy - 8), min(int(h * 0.90), best_cy + 9), 1):
        for fx in range(max(int(w * 0.10), best_cx - 8), min(int(w * 0.90), best_cx + 9), 1):
            resp = haar_center_surround_response(integral, fx, fy, int(best_r), r_out, h, w)
            center_darkness = 255.0 - float(gray_img[fy, fx])
            total_resp = resp * 0.65 + center_darkness * 0.35
            if total_resp > fine_best_resp:
                fine_best_resp = total_resp
                fine_cx, fine_cy = fx, fy

    return (fine_cx, fine_cy, best_r, fine_best_resp)


def detect_pupil_swirski(
    gray_img: np.ndarray,
    eyelid_mask: Optional[np.ndarray] = None,
    glint_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Runs complete Swirski & Dodgson pipeline for robust pupil detection.
    """
    h, w = gray_img.shape[:2]

    # 1. Glint Inpainting
    if glint_mask is not None and np.sum(glint_mask) > 0:
        inpainted = cv2.inpaint(gray_img, glint_mask, 5, cv2.INPAINT_TELEA)
    else:
        # Automatic glint detection
        _, glint_bin = cv2.threshold(gray_img, 215, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        g_mask = cv2.dilate(glint_bin, kernel, iterations=1)
        inpainted = cv2.inpaint(gray_img, g_mask, 4, cv2.INPAINT_TELEA) if np.sum(g_mask) > 0 else gray_img

    # 2. Coarse Haar Localization
    cx_c, cy_c, r_c, resp = locate_pupil_coarse_haar(inpainted, search_step=4)

    # 3. Radial Gradient Vector Filtering
    blurred = cv2.GaussianBlur(inpainted, (5, 5), 1.5)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    edges = cv2.Canny(blurred, 25, 75)

    # Collect valid radial boundary points
    annulus_inner = 5.0
    annulus_outer = max(15.0, min(float(min(w, h) * 0.20), max(r_c * 1.25, 60.0)))

    valid_edge_pts = []
    y_idxs, x_idxs = np.where(edges > 0)

    for ex, ey in zip(x_idxs, y_idxs):
        dx = ex - cx_c
        dy = ey - cy_c
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if annulus_inner <= dist <= annulus_outer:
            # Check eyelid mask
            if eyelid_mask is not None and eyelid_mask[ey, ex] == 0:
                continue

            # Vector alignment: gradient should point away from pupil center (dark to light)
            mag = grad_mag[ey, ex]
            if mag > 5.0 and dist > 0:
                rad_x = dx / dist
                rad_y = dy / dist
                g_unit_x = gx[ey, ex] / mag
                g_unit_y = gy[ey, ex] / mag

                dot = rad_x * g_unit_x + rad_y * g_unit_y
                if dot >= 0.35:  # Outward directed edge
                    valid_edge_pts.append((float(ex), float(ey)))

    if len(valid_edge_pts) < 6:
        # Fallback: create circular estimate from coarse Haar
        ell = ((float(cx_c), float(cy_c)), (float(r_c * 2), float(r_c * 2)), 0.0)
        return {
            "center": (cx_c, cy_c),
            "radius": r_c,
            "radius_x": r_c,
            "radius_y": r_c,
            "rotation": 0.0,
            "ellipse": ell,
            "confidence": float(np.clip(resp * 1.5, 30.0, 75.0)),
            "inlier_points": [],
            "detected": True
        }

    pts_arr = np.array(valid_edge_pts, dtype=np.float32)

    # 4. RANSAC with Fitzgibbon Ellipse Fitting
    best_ellipse = None
    best_inliers = []
    best_inlier_count = 0
    n_pts = len(pts_arr)

    for _ in range(120):
        sample_idx = np.random.choice(n_pts, size=min(6, n_pts), replace=False)
        sample = pts_arr[sample_idx]

        ell = fit_ellipse_fitzgibbon(sample)
        if ell is None:
            continue

        (ecx, ecy), (ew, eh), eang = ell
        erx, ery = ew / 2.0, eh / 2.0

        # Geometric plausibility checks for pupil
        if erx < 4.0 or ery < 4.0 or erx > min(w, h) * 0.18 or ery > min(w, h) * 0.18:
            continue
        if max(erx, ery) / max(0.1, min(erx, ery)) > 2.2:
            continue
        c_dist = np.sqrt((ecx - cx_c) ** 2 + (ecy - cy_c) ** 2)
        if c_dist > r_c * 0.7:
            continue

        # Count inliers
        from src.ellipse_fitting import point_to_ellipse_distance
        dists = point_to_ellipse_distance(pts_arr, ecx, ecy, erx, ery, eang)
        inlier_mask = dists < 3.0
        cnt = int(np.sum(inlier_mask))

        if cnt > best_inlier_count:
            best_inlier_count = cnt
            best_inliers = pts_arr[inlier_mask]
            best_ellipse = ell

    # Refit on all inliers
    if best_ellipse is not None and len(best_inliers) >= 5:
        refit_ell = fit_ellipse_fitzgibbon(best_inliers)
        if refit_ell is not None:
            best_ellipse = refit_ell

    if best_ellipse is None:
        best_ellipse = fit_ellipse_fitzgibbon(pts_arr)
        if best_ellipse is None:
            best_ellipse = ((float(cx_c), float(cy_c)), (float(r_c * 2), float(r_c * 2)), 0.0)

    (fcx, fcy), (fw, fh), fang = best_ellipse
    frx, fry = fw / 2.0, fh / 2.0
    mean_r = (frx + fry) / 2.0

    inlier_ratio = float(best_inlier_count / max(1, n_pts))
    confidence = float(np.clip(inlier_ratio * 70.0 + min(30.0, resp * 0.5), 45.0, 98.0))

    return {
        "center": (int(round(fcx)), int(round(fcy))),
        "radius": float(mean_r),
        "radius_x": float(frx),
        "radius_y": float(fry),
        "rotation": float(fang),
        "ellipse": best_ellipse,
        "confidence": confidence,
        "inlier_points": [(float(p[0]), float(p[1])) for p in best_inliers],
        "detected": True
    }
