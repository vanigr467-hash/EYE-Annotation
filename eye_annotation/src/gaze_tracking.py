import cv2
import numpy as np
from typing import Dict, Any, Tuple

def calculate_gaze_vector(
    pupil_center: Tuple[int, int],
    iris_center: Tuple[int, int],
    iris_radius: float
) -> Dict[str, Any]:
    """
    Calculates 2D Eye Gaze Vector, Gaze Angle, Offset Ratio, and Cardinal Direction.
    
    Vector G = (X_pupil - X_iris, Y_pupil - Y_iris)
    """
    px, py = pupil_center
    ix, iy = iris_center
    ir = max(1.0, iris_radius)

    dx = px - ix
    dy = py - iy
    distance = np.sqrt(dx**2 + dy**2)
    normalized_offset = (distance / ir) * 100.0

    angle_deg = np.degrees(np.arctan2(dy, dx))

    # Determine Cardinal Gaze Direction
    if distance < (ir * 0.08):
        direction_label = "Direct / Center"
    else:
        if -22.5 <= angle_deg < 22.5:
            direction_label = "Looking Right"
        elif 22.5 <= angle_deg < 67.5:
            direction_label = "Looking Down-Right"
        elif 67.5 <= angle_deg < 112.5:
            direction_label = "Looking Down"
        elif 112.5 <= angle_deg < 157.5:
            direction_label = "Looking Down-Left"
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            direction_label = "Looking Left"
        elif -157.5 <= angle_deg < -112.5:
            direction_label = "Looking Up-Left"
        elif -112.5 <= angle_deg < -67.5:
            direction_label = "Looking Up"
        else: # -67.5 to -22.5
            direction_label = "Looking Up-Right"

    return {
        "dx": dx,
        "dy": dy,
        "distance_px": round(distance, 2),
        "normalized_offset_pct": round(normalized_offset, 2),
        "gaze_angle_deg": round(angle_deg, 1),
        "direction_label": direction_label
    }

def draw_gaze_vector_arrow(
    image_rgb: np.ndarray,
    pupil_center: Tuple[int, int],
    iris_center: Tuple[int, int],
    gaze_info: Dict[str, Any],
    scale_factor: float = 3.0
) -> np.ndarray:
    """Renders 2D Gaze Direction Arrow overlay from Iris center through Pupil center."""
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]
    ix, iy = iris_center
    px, py = pupil_center

    dx = gaze_info["dx"]
    dy = gaze_info["dy"]

    # Extended arrow tip
    tip_x = int(round(ix + dx * scale_factor))
    tip_y = int(round(iy + dy * scale_factor))

    # Arrow line color (Yellow / Green)
    arrow_color = (0, 255, 128) # Bright Mint Green

    thickness = max(2, int(w * 0.005))
    cv2.arrowedLine(
        annotated,
        (ix, iy), (tip_x, tip_y),
        arrow_color, thickness,
        tipLength=0.3
    )

    # Draw gaze target dot at tip
    cv2.circle(annotated, (tip_x, tip_y), max(4, int(w * 0.008)), (255, 255, 0), -1)

    return annotated
