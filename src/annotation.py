import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from src.pupil_detection import PupilResult
from src.iris_detection import IrisResult

COLOR_SCHEMES = {
    "Clinical Blueprint": {
        "pupil_line": (0, 220, 255),    # Cyan
        "pupil_point": (0, 255, 255),   # Bright Yellow/Cyan
        "pupil_center": (255, 50, 50),   # Red
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

def generate_boundary_points(
    center: Tuple[int, int],
    radius: float,
    num_points: int = 32
) -> List[Tuple[int, int]]:
    """Generates num_points evenly spaced Cartesian (X, Y) coordinates along circumference."""
    if radius <= 0:
        return []
    
    points = []
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    for a in angles:
        px = int(round(cx + radius * np.cos(a)))
        py = int(round(cy + radius * np.sin(a)))
        points.append((px, py))
        
    return points

def draw_automatic_annotations(
    image_rgb: np.ndarray,
    pupil_res: PupilResult,
    iris_res: IrisResult,
    num_points: int = 32,
    theme: str = "Clinical Blueprint",
    show_pupil: bool = True,
    show_iris: bool = True,
    show_centers: bool = True,
    show_points: bool = True,
    show_labels: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders automatic pupil and iris annotations onto the RGB image.
    
    Returns:
        annotated_rgb: np.ndarray
        points_data: dict containing pupil_points and iris_points lists
    """
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]
    colors = COLOR_SCHEMES.get(theme, COLOR_SCHEMES["Clinical Blueprint"])

    pupil_pts = []
    iris_pts = []

    # Scale line thickness based on image size
    thickness = max(2, int(w * 0.004))
    point_radius = max(3, int(w * 0.006))

    # 1. Annotate Pupil
    if pupil_res.detected and show_pupil and pupil_res.radius > 0:
        px, py = pupil_res.center
        pr = int(pupil_res.radius)

        # Draw continuous pupil boundary circle/ellipse
        if pupil_res.ellipse is not None:
            ((ecx, ecy), (e_a, e_b), angle) = pupil_res.ellipse
            cv2.ellipse(
                annotated,
                ((int(ecx), int(ecy)), (int(e_a), int(e_b)), angle),
                colors["pupil_line"],
                thickness
            )
        else:
            cv2.circle(annotated, (px, py), pr, colors["pupil_line"], thickness)

        # Generate evenly spaced boundary points
        pupil_pts = generate_boundary_points((px, py), pupil_res.radius, num_points)

        if show_points:
            for idx, (pt_x, pt_y) in enumerate(pupil_pts):
                cv2.circle(annotated, (pt_x, pt_y), point_radius, colors["pupil_point"], -1)
                if show_labels and (idx % max(1, num_points // 8) == 0):
                    cv2.putText(
                        annotated, f"P{idx+1}", (pt_x + 3, pt_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )

        # Draw Pupil Center Crosshair
        if show_centers:
            cv2.drawMarker(
                annotated, (px, py), colors["pupil_center"],
                cv2.MARKER_CROSS, markerSize=int(pr * 0.4), thickness=thickness
            )
            cv2.circle(annotated, (px, py), 3, colors["pupil_center"], -1)

    # 2. Annotate Iris
    if iris_res.detected and show_iris and iris_res.radius > 0:
        ix, iy = iris_res.center
        ir = int(iris_res.radius)

        # Draw iris boundary circle
        cv2.circle(annotated, (ix, iy), ir, colors["iris_line"], thickness)

        # Generate iris boundary points
        iris_pts = generate_boundary_points((ix, iy), iris_res.radius, num_points)

        if show_points:
            for idx, (pt_x, pt_y) in enumerate(iris_pts):
                cv2.circle(annotated, (pt_x, pt_y), point_radius, colors["iris_point"], -1)
                if show_labels and (idx % max(1, num_points // 8) == 0):
                    cv2.putText(
                        annotated, f"I{idx+1}", (pt_x + 4, pt_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )

        # Draw Iris Center Crosshair
        if show_centers:
            cv2.drawMarker(
                annotated, (ix, iy), colors["iris_center"],
                cv2.MARKER_TILTED_CROSS, markerSize=int(ir * 0.3), thickness=thickness
            )

    points_data = {
        "num_points": num_points,
        "pupil_points": pupil_pts,
        "iris_points": iris_pts
    }

    return annotated, points_data
