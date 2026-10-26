"""
Annotation Module for Eye Boundary Visualization and Clean Keypoint Placement.

Renders:
- Clean, mathematically smooth Pupil and Iris boundaries (Circle or Ellipse)
- N Configurable annotation points (8, 16, 32, 64, 128) lying smoothly on the boundary
- Iris Boundary Evidence Debug Mode (Inlier vs Outlier points)
- Eyelid occlusion curves
- 2D Eye Gaze direction vector arrow
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from src.ellipse_fitting import generate_ellipse_points, refine_points_to_edge
from src.gaze_tracking import calculate_gaze_vector, draw_gaze_vector_arrow


COLOR_SCHEMES = {
    "Clinical Blueprint": {
        "pupil_line": (0, 220, 255),    # Cyan
        "pupil_point": (0, 255, 255),   # Bright Cyan
        "pupil_center": (255, 50, 50),  # Red
        "iris_line": (255, 105, 180),   # Magenta / Hot Pink
        "iris_point": (255, 215, 0),    # Gold
        "iris_center": (0, 255, 0)      # Green
    },
    "Neon Cyberpunk": {
        "pupil_line": (50, 255, 50),    # Neon Green
        "pupil_point": (255, 255, 0),   # Yellow
        "pupil_center": (255, 0, 255),  # Magenta
        "iris_line": (0, 191, 255),     # Deep Sky Blue
        "iris_point": (255, 140, 0),    # Dark Orange
        "iris_center": (255, 255, 255)  # White
    },
    "High Contrast": {
        "pupil_line": (255, 0, 0),      # Pure Red
        "pupil_point": (255, 255, 0),   # Pure Yellow
        "pupil_center": (255, 255, 255),# White
        "iris_line": (0, 0, 255),       # Pure Blue
        "iris_point": (0, 255, 255),    # Pure Cyan
        "iris_center": (255, 255, 255)  # White
    }
}

_COLOR_INLIER = (0, 255, 0)          # Green
_COLOR_OUTLIER = (255, 60, 60)       # Red
_COLOR_DEBUG_ELLIPSE = (0, 255, 255) # Cyan


def generate_boundary_points(
    center: Tuple[int, int],
    radius: float,
    num_points: int = 64
) -> List[Tuple[int, int]]:
    """Generates num_points evenly spaced points along a circular perimeter."""
    if radius <= 0 or num_points <= 0:
        return []
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = []
    for a in angles:
        px = int(round(cx + radius * np.cos(a)))
        py = int(round(cy + radius * np.sin(a)))
        pts.append((px, py))
    return pts


def generate_boundary_points_for_result(
    result: Any,
    num_points: int = 64,
    gradient_img: Optional[np.ndarray] = None,
    refine_to_edge: bool = False
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Generates N boundary points for a PupilResult or IrisResult.
    By default, points lie cleanly on the smooth fitted ellipse/circle.
    
    Returns:
        (points: List[(x, y)], quality_dict)
    """
    if result is None or not getattr(result, "detected", False) or result.radius <= 0:
        return [], {
            "total_points": 0,
            "valid_points": 0,
            "inlier_pct": 0.0,
            "mean_edge_strength": 0.0,
            "edge_strengths": []
        }

    center = result.center
    radius = result.radius
    ellipse = getattr(result, "ellipse", None)

    if ellipse is not None:
        (ecx, ecy), (ew, eh), eang = ellipse
        raw_pts = generate_ellipse_points(ecx, ecy, ew / 2.0, eh / 2.0, eang, num_points)
    else:
        raw_pts = generate_boundary_points(center, radius, num_points)

    if refine_to_edge and gradient_img is not None and len(raw_pts) > 0:
        refined_pts, strengths = refine_points_to_edge(
            raw_pts, gradient_img, search_radius=3, ellipse_center=center
        )
        valid_count = sum(1 for s in strengths if s > 5.0)
        inlier_pct = round((valid_count / float(max(1, len(raw_pts)))) * 100.0, 1)
        mean_edge = round(float(np.mean(strengths)), 1) if strengths else 0.0
        quality = {
            "total_points": len(refined_pts),
            "valid_points": valid_count,
            "inlier_pct": inlier_pct,
            "mean_edge_strength": mean_edge,
            "edge_strengths": strengths
        }
        return refined_pts, quality
    else:
        # Sample edge strengths at the clean ellipse points for diagnostic reporting
        strengths = []
        if gradient_img is not None:
            gh, gw = gradient_img.shape[:2]
            for px, py in raw_pts:
                if 0 <= px < gw and 0 <= py < gh:
                    strengths.append(float(gradient_img[py, px]))
                else:
                    strengths.append(0.0)
            valid_count = sum(1 for s in strengths if s > 5.0)
            inlier_pct = round((valid_count / float(max(1, len(raw_pts)))) * 100.0, 1)
            mean_edge = round(float(np.mean(strengths)), 1) if strengths else 0.0
        else:
            valid_count = len(raw_pts)
            inlier_pct = 100.0
            mean_edge = 0.0

        return raw_pts, {
            "total_points": len(raw_pts),
            "valid_points": valid_count,
            "inlier_pct": inlier_pct,
            "mean_edge_strength": mean_edge,
            "edge_strengths": strengths
        }


def _draw_boundary_badge(
    image: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (15, 23, 42),
    font_scale: float = 0.42,
    thickness: int = 1
) -> None:
    """Renders high-contrast pill text badge with background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    ih, iw = image.shape[:2]
    bx1 = max(4, min(x, iw - tw - 12))
    by1 = max(4, min(y - th - 6, ih - th - 12))
    bx2 = bx1 + tw + 8
    by2 = by1 + th + 8

    # Background badge with slight transparency
    overlay = image.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), text_color, 1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Clean text with subtle outline
    cv2.putText(image, text, (bx1 + 4, by1 + th + 3), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (bx1 + 4, by1 + th + 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_automatic_annotations(
    image_rgb: np.ndarray,
    pupil_res: Any,
    iris_res: Any,
    num_points: int = 64,
    theme: str = "Clinical Blueprint",
    show_pupil: bool = True,
    show_iris: bool = True,
    show_centers: bool = True,
    show_points: bool = True,
    show_labels: bool = False,
    show_gaze_arrow: bool = False,
    show_boundary_evidence: bool = False,
    eyelid_mask_obj: Optional[Any] = None,
    refine_points: bool = False,
    show_values: bool = False,
    boundary_style: str = "Circles / Ellipses",
    values_format: str = "Radius & Center (r, x, y)"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders automatic pupil and iris annotations, smooth keypoints, boundary measurement values,
    and debug layers onto RGB image.
    """
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]
    colors = COLOR_SCHEMES.get(theme, COLOR_SCHEMES["Clinical Blueprint"])

    style_lower = boundary_style.lower()
    draw_curves = "numbers only" not in style_lower and "dots only" not in style_lower
    effective_show_points = show_points or "dots" in style_lower or "full" in style_lower
    effective_show_values = show_values or "number" in style_lower or "value" in style_lower or "full" in style_lower
    effective_show_centers = show_centers or "number" in style_lower

    thickness = max(2, int(w * 0.004))
    point_radius = max(3, int(w * 0.006))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if len(image_rgb.shape) == 3 else image_rgb
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    pupil_pts, pupil_q = generate_boundary_points_for_result(
        pupil_res, num_points, grad_mag, refine_to_edge=refine_points
    )
    iris_pts, iris_q = generate_boundary_points_for_result(
        iris_res, num_points, grad_mag, refine_to_edge=refine_points
    )

    # 1. Annotate Pupil (Smooth clean line, points & values)
    if pupil_res and getattr(pupil_res, "detected", False) and show_pupil and pupil_res.radius > 0:
        px, py = pupil_res.center
        pr = int(round(pupil_res.radius))
        prx = float(getattr(pupil_res, "radius_x", pupil_res.radius))
        pry = float(getattr(pupil_res, "radius_y", pupil_res.radius))

        if draw_curves:
            if getattr(pupil_res, "ellipse", None) is not None:
                ((ecx, ecy), (ew, eh), angle) = pupil_res.ellipse
                cv2.ellipse(
                    annotated,
                    (int(round(ecx)), int(round(ecy))),
                    (int(round(ew / 2.0)), int(round(eh / 2.0))),
                    angle, 0, 360,
                    colors["pupil_line"],
                    thickness
                )
            else:
                cv2.circle(annotated, (px, py), pr, colors["pupil_line"], thickness)

        if effective_show_points:
            for idx, (pt_x, pt_y) in enumerate(pupil_pts):
                cv2.circle(annotated, (pt_x, pt_y), point_radius, colors["pupil_point"], -1)
                if show_labels and (idx % max(1, num_points // 8) == 0):
                    cv2.putText(
                        annotated, f"P{idx+1}", (pt_x + 3, pt_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )

        if effective_show_centers:
            cv2.drawMarker(
                annotated, (px, py), colors["pupil_center"],
                cv2.MARKER_CROSS, markerSize=max(8, int(pr * 0.4)), thickness=thickness
            )
            cv2.circle(annotated, (px, py), 2, colors["pupil_center"], -1)

        if effective_show_values:
            if values_format.startswith("Radius only"):
                p_text = f"Pupil r={pupil_res.radius:.1f}px"
            elif values_format.startswith("Full Geometry"):
                p_text = f"Pupil ({px},{py}) rx:{prx:.1f} ry:{pry:.1f}"
            else:
                p_text = f"Pupil r={pupil_res.radius:.1f}px ({px},{py})"
            badge_y = max(18, int(py - pry - 8))
            _draw_boundary_badge(annotated, p_text, (max(5, px - int(prx * 0.6)), badge_y), colors["pupil_line"])

    # 2. Annotate Iris (Smooth clean line, points & values)
    if iris_res and getattr(iris_res, "detected", False) and show_iris and iris_res.radius > 0:
        ix, iy = iris_res.center
        irx = float(getattr(iris_res, "radius_x", iris_res.radius))
        iry = float(getattr(iris_res, "radius_y", iris_res.radius))

        if draw_curves:
            if getattr(iris_res, "ellipse", None) is not None:
                ((ecx, ecy), (ew, eh), angle) = iris_res.ellipse
                cv2.ellipse(
                    annotated,
                    (int(round(ecx)), int(round(ecy))),
                    (int(round(ew / 2.0)), int(round(eh / 2.0))),
                    angle, 0, 360,
                    colors["iris_line"],
                    thickness
                )
            else:
                cv2.circle(annotated, (ix, iy), int(round(iris_res.radius)), colors["iris_line"], thickness)

        if effective_show_points:
            for idx, (pt_x, pt_y) in enumerate(iris_pts):
                cv2.circle(annotated, (pt_x, pt_y), point_radius, colors["iris_point"], -1)
                if show_labels and (idx % max(1, num_points // 8) == 0):
                    cv2.putText(
                        annotated, f"I{idx+1}", (pt_x + 4, pt_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )

        if effective_show_centers:
            cv2.drawMarker(
                annotated, (ix, iy), colors["iris_center"],
                cv2.MARKER_TILTED_CROSS, markerSize=max(10, int(iris_res.radius * 0.2)), thickness=thickness
            )

        if effective_show_values:
            if values_format.startswith("Radius only"):
                i_text = f"Iris r={iris_res.radius:.1f}px"
            elif values_format.startswith("Full Geometry"):
                i_text = f"Iris ({ix},{iy}) rx:{irx:.1f} ry:{iry:.1f}"
            else:
                i_text = f"Iris r={iris_res.radius:.1f}px ({ix},{iy})"
            badge_y = max(18, int(iy - iry - 8))
            _draw_boundary_badge(annotated, i_text, (max(5, ix - int(irx * 0.6)), badge_y), colors["iris_line"])

    # 3. Iris Boundary Evidence Debug Mode
    if show_boundary_evidence and iris_res:
        annotated = _draw_boundary_evidence(annotated, iris_res, w)

    # 4. Eyelid Curves (if available and debug on)
    if show_boundary_evidence and eyelid_mask_obj is not None:
        annotated = _draw_eyelid_curves(annotated, eyelid_mask_obj)

    # 5. Gaze Vector Arrow
    if (show_gaze_arrow and pupil_res and iris_res and
        getattr(pupil_res, "detected", False) and getattr(iris_res, "detected", False) and
        iris_res.radius > 0):
        gaze_info = calculate_gaze_vector(pupil_res.center, iris_res.center, iris_res.radius)
        annotated = draw_gaze_vector_arrow(annotated, pupil_res.center, iris_res.center, gaze_info)

    points_data = {
        "num_points": num_points,
        "pupil_points": pupil_pts,
        "pupil_quality": pupil_q,
        "iris_points": iris_pts,
        "iris_quality": iris_q
    }

    return annotated, points_data


def _draw_boundary_evidence(
    image: np.ndarray,
    iris_res: Any,
    img_width: int
) -> np.ndarray:
    """Renders debug markers for candidate, inlier, and outlier boundary points."""
    annotated = image.copy()
    dbg_r = max(4, int(img_width * 0.007))

    for pt in getattr(iris_res, "outlier_points", []):
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(annotated, (px, py), dbg_r, _COLOR_OUTLIER, -1)
        arm = dbg_r + 1
        cv2.line(annotated, (px - arm, py - arm), (px + arm, py + arm), _COLOR_OUTLIER, 1)
        cv2.line(annotated, (px - arm, py + arm), (px + arm, py - arm), _COLOR_OUTLIER, 1)

    for pt in getattr(iris_res, "inlier_points", []):
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(annotated, (px, py), dbg_r, _COLOR_INLIER, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, img_width * 0.0007)
    cv2.putText(annotated, "Iris Evidence:", (10, 22), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(annotated, (15, 40), 5, _COLOR_INLIER, -1)
    cv2.putText(annotated, f"Inliers ({len(getattr(iris_res, 'inlier_points', []))})", (26, 44), font, font_scale * 0.9, _COLOR_INLIER, 1, cv2.LINE_AA)
    cv2.circle(annotated, (15, 60), 5, _COLOR_OUTLIER, -1)
    cv2.putText(annotated, f"Outliers ({len(getattr(iris_res, 'outlier_points', []))})", (26, 64), font, font_scale * 0.9, _COLOR_OUTLIER, 1, cv2.LINE_AA)

    return annotated


def _draw_eyelid_curves(image: np.ndarray, eyelid_obj: Any) -> np.ndarray:
    """Draws upper/lower eyelid parabolic curves."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    if getattr(eyelid_obj, "upper_curve_y", None) is not None:
        y_vals = eyelid_obj.upper_curve_y
        pts = []
        for x_idx, y_val in enumerate(y_vals):
            if 0 <= y_val < h:
                pts.append((x_idx, int(round(y_val))))
        if len(pts) > 2:
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)

    if getattr(eyelid_obj, "lower_curve_y", None) is not None:
        y_vals = eyelid_obj.lower_curve_y
        pts = []
        for x_idx, y_val in enumerate(y_vals):
            if 0 <= y_val < h:
                pts.append((x_idx, int(round(y_val))))
        if len(pts) > 2:
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)

    return annotated


def draw_annotation_only_mask(
    shape: Tuple[int, ...],
    pupil_res: Any,
    iris_res: Any,
    num_points: int = 64,
    theme: str = "Clinical Blueprint",
    show_pupil: bool = True,
    show_iris: bool = True,
    show_centers: bool = True,
    show_points: bool = True,
    show_labels: bool = False,
    show_values: bool = False,
    boundary_style: str = "Circles / Ellipses",
    values_format: str = "Radius & Center (r, x, y)",
    eyelid_mask_obj: Optional[Any] = None
) -> np.ndarray:
    """Renders pure annotations onto a clean black canvas mask, returning the mask image as np.ndarray."""
    h, w = shape[:2]
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask_canvas, _ = draw_automatic_annotations(
        black_canvas,
        pupil_res,
        iris_res,
        num_points=num_points,
        theme=theme,
        show_pupil=show_pupil,
        show_iris=show_iris,
        show_centers=show_centers,
        show_points=show_points,
        show_labels=show_labels,
        show_values=show_values,
        boundary_style=boundary_style,
        values_format=values_format,
        eyelid_mask_obj=eyelid_mask_obj
    )
    return mask_canvas
