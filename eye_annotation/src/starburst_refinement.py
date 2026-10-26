import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

def refine_pupil_boundary_starburst(
    gray_img: np.ndarray,
    initial_center: Tuple[int, int],
    initial_radius: float,
    num_rays: int = 48,
    ransac_iterations: int = 100,
    inlier_thresh: float = 3.5
) -> Dict[str, Any]:
    """
    Sub-pixel boundary point refinement using Starburst Radial Gradient Ray-Tracing 
    and RANSAC outlier pruning.
    
    1. Casts num_rays outwards from initial_center.
    2. Finds peak negative intensity gradient along each ray (pupil perimeter edge).
    3. Uses RANSAC to prune eyelash/shadow/glint outlier points.
    4. Fits refined circle or ellipse on inlier points.
    
    Returns dict:
        refined_center: (x, y)
        refined_radius: float
        ellipse: fitted ellipse tuple or None
        inlier_points: List[(x, y)]
        outlier_points: List[(x, y)]
        confidence_boost: float
    """
    h, w = gray_img.shape[:2]
    cx, cy = initial_center
    r_est = max(5.0, initial_radius)

    if r_est <= 3 or cx <= 10 or cx >= w - 10 or cy <= 10 or cy >= h - 10:
        return {
            "refined_center": initial_center,
            "refined_radius": initial_radius,
            "ellipse": None,
            "inlier_points": [],
            "outlier_points": [],
            "confidence_boost": 0.0
        }

    # Apply Gaussian smoothing for gradient computation
    smoothed = cv2.GaussianBlur(gray_img, (5, 5), 1.2)

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    candidate_points = []

    min_r_search = max(3.0, r_est * 0.45)
    max_r_search = min(min(w, h) * 0.45, r_est * 1.65)

    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        r_samples = np.linspace(min_r_search, max_r_search, int(max_r_search - min_r_search + 1))
        sample_coords = []
        intensities = []

        for r in r_samples:
            px = cx + r * cos_a
            py = cy + r * sin_a

            if 0 <= px < w - 1 and 0 <= py < h - 1:
                # Sub-pixel bilinear interpolation
                x0, y0 = int(px), int(py)
                dx, dy = px - x0, py - y0
                val = (1 - dx) * (1 - dy) * smoothed[y0, x0] + \
                      dx * (1 - dy) * smoothed[y0, x0 + 1] + \
                      (1 - dx) * dy * smoothed[y0 + 1, x0] + \
                      dx * dy * smoothed[y0 + 1, x0 + 1]
                intensities.append(val)
                sample_coords.append((px, py))

        if len(intensities) >= 5:
            # Radial intensity gradient: dI/dr (Pupil edge is dark to light -> max positive dI/dr)
            grads = np.diff(intensities)
            max_grad_idx = int(np.argmax(grads))
            
            # Require significant gradient step
            if grads[max_grad_idx] > 3.0:
                edge_pt = sample_coords[max_grad_idx]
                candidate_points.append(edge_pt)

    if len(candidate_points) < 6:
        return {
            "refined_center": initial_center,
            "refined_radius": initial_radius,
            "ellipse": None,
            "inlier_points": candidate_points,
            "outlier_points": [],
            "confidence_boost": 0.0
        }

    # RANSAC Circle / Ellipse Fitting
    best_inliers = []
    best_center = initial_center
    best_radius = initial_radius
    best_ellipse = None

    pts_arr = np.array(candidate_points, dtype=np.float32)
    n_pts = len(pts_arr)

    for _ in range(ransac_iterations):
        sample_indices = np.random.choice(n_pts, size=min(5, n_pts), replace=False)
        sample_pts = pts_arr[sample_indices]

        # Estimate circle from sample
        (sample_cx, sample_cy), sample_r = cv2.minEnclosingCircle(sample_pts)

        if sample_r <= 0:
            continue

        # Measure distances from all points to sample circle
        dists = np.abs(np.sqrt((pts_arr[:, 0] - sample_cx)**2 + (pts_arr[:, 1] - sample_cy)**2) - sample_r)
        inliers_mask = dists < inlier_thresh

        if np.sum(inliers_mask) > len(best_inliers):
            best_inliers = [candidate_points[i] for i in range(n_pts) if inliers_mask[i]]
            best_center = (int(round(sample_cx)), int(round(sample_cy)))
            best_radius = float(sample_r)

    inlier_pts_arr = np.array(best_inliers, dtype=np.float32) if best_inliers else pts_arr

    if len(inlier_pts_arr) >= 5:
        try:
            ellipse_param = cv2.fitEllipse(inlier_pts_arr)
            ((ecx, ecy), (ea, eb), e_angle) = ellipse_param
            best_ellipse = ((ecx, ecy), (ea, eb), e_angle)
            best_center = (int(round(ecx)), int(round(ecy)))
            best_radius = float((ea + eb) / 4.0)
        except Exception:
            (bcx, bcy), br = cv2.minEnclosingCircle(inlier_pts_arr)
            best_center = (int(round(bcx)), int(round(bcy)))
            best_radius = float(br)

    outlier_points = [pt for pt in candidate_points if pt not in best_inliers]
    confidence_boost = min(15.0, (len(best_inliers) / float(num_rays)) * 20.0)

    return {
        "refined_center": best_center,
        "refined_radius": best_radius,
        "ellipse": best_ellipse,
        "inlier_points": best_inliers,
        "outlier_points": outlier_points,
        "confidence_boost": round(confidence_boost, 1)
    }
