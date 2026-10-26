"""
Ellipse Editor Component for manual annotation.
Provides interactive ellipse parameter controls and real-time preview overlay.
"""

import cv2
import numpy as np
import streamlit as st
from typing import Dict, Optional, Tuple


def draw_ellipse_overlay(
    image_rgb: np.ndarray,
    cx: float, cy: float,
    rx: float, ry: float,
    angle: float = 0.0,
    label_type: str = "pupil",
    show_center: bool = True,
    show_axes: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw an ellipse overlay on the image for preview."""
    overlay = image_rgb.copy()
    center = (int(round(cx)), int(round(cy)))
    axes = (int(round(rx)), int(round(ry)))

    # Color based on label type
    if label_type == "pupil":
        color = (0, 255, 0)  # Green
        center_color = (0, 200, 0)
    else:
        color = (0, 180, 255)  # Orange
        center_color = (0, 140, 200)

    # Draw ellipse
    if rx > 0 and ry > 0:
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)

    # Draw center crosshair
    if show_center:
        size = 8
        cv2.line(overlay, (center[0] - size, center[1]), (center[0] + size, center[1]),
                 center_color, 1, cv2.LINE_AA)
        cv2.line(overlay, (center[0], center[1] - size), (center[0], center[1] + size),
                 center_color, 1, cv2.LINE_AA)

    # Draw semi-axes lines
    if show_axes and rx > 0 and ry > 0:
        angle_rad = np.radians(angle)
        # Major axis endpoints
        dx = rx * np.cos(angle_rad)
        dy = rx * np.sin(angle_rad)
        cv2.line(overlay, center, (int(cx + dx), int(cy + dy)), color, 1, cv2.LINE_AA)
        # Minor axis endpoints
        dx2 = ry * np.cos(angle_rad + np.pi / 2)
        dy2 = ry * np.sin(angle_rad + np.pi / 2)
        cv2.line(overlay, center, (int(cx + dx2), int(cy + dy2)), color, 1, cv2.LINE_AA)

    # Label
    label = label_type.upper()
    cv2.putText(overlay, label, (center[0] + 12, center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return overlay


def ellipse_editor_widget(
    image_rgb: np.ndarray,
    label_type: str = "pupil",
    initial_params: Optional[Dict[str, float]] = None,
    key_prefix: str = "editor",
) -> Dict[str, float]:
    """
    Streamlit widget for editing ellipse parameters with real-time preview.

    Args:
        image_rgb: The source image (RGB format)
        label_type: 'pupil' or 'iris'
        initial_params: Optional initial ellipse parameters {cx, cy, rx, ry, angle}
        key_prefix: Unique key prefix for Streamlit widgets

    Returns:
        Dictionary with edited ellipse parameters {cx, cy, rx, ry, angle}
    """
    h, w = image_rgb.shape[:2]

    # Default initial values
    defaults = {
        "cx": w / 2.0,
        "cy": h / 2.0,
        "rx": min(w, h) * 0.1,
        "ry": min(w, h) * 0.1,
        "angle": 0.0,
    }
    if initial_params:
        defaults.update(initial_params)

    color_label = "🟢 Pupil" if label_type == "pupil" else "🟠 Iris"
    st.markdown(f"#### {color_label} Ellipse Parameters")

    col1, col2 = st.columns(2)
    with col1:
        cx = st.slider(
            "Center X", 0.0, float(w), defaults["cx"],
            step=0.5, key=f"{key_prefix}_{label_type}_cx"
        )
        rx = st.slider(
            "Radius X (semi-major)", 1.0, float(w) / 2, defaults["rx"],
            step=0.5, key=f"{key_prefix}_{label_type}_rx"
        )
    with col2:
        cy = st.slider(
            "Center Y", 0.0, float(h), defaults["cy"],
            step=0.5, key=f"{key_prefix}_{label_type}_cy"
        )
        ry = st.slider(
            "Radius Y (semi-minor)", 1.0, float(h) / 2, defaults["ry"],
            step=0.5, key=f"{key_prefix}_{label_type}_ry"
        )

    angle = st.slider(
        "Rotation Angle (°)", -180.0, 180.0, defaults["angle"],
        step=1.0, key=f"{key_prefix}_{label_type}_angle"
    )

    return {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle}


def render_annotation_preview(
    image_rgb: np.ndarray,
    pupil_params: Optional[Dict[str, float]] = None,
    iris_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Render image with both pupil and iris ellipse overlays."""
    result = image_rgb.copy()

    if iris_params:
        result = draw_ellipse_overlay(
            result,
            iris_params["cx"], iris_params["cy"],
            iris_params["rx"], iris_params["ry"],
            iris_params.get("angle", 0.0),
            label_type="iris",
            thickness=2
        )

    if pupil_params:
        result = draw_ellipse_overlay(
            result,
            pupil_params["cx"], pupil_params["cy"],
            pupil_params["rx"], pupil_params["ry"],
            pupil_params.get("angle", 0.0),
            label_type="pupil",
            thickness=2
        )

    return result
