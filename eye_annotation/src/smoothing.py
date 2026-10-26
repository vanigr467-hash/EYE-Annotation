"""
Temporal smoothing via Exponential Moving Average (EMA) for pupil and iris
parameters across video frames, including iris ellipse axes and rotation.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

SMOOTHING_ALPHAS = {
    "OFF": 1.0,
    "LOW": 0.75,
    "MEDIUM": 0.50,
    "HIGH": 0.25
}


class TemporalSmoother:
    """
    Applies Exponential Moving Average (EMA) temporal smoothing across video frames
    to prevent flickering and sudden boundary jumps.
    """
    def __init__(self, mode: str = "MEDIUM"):
        self.alpha = SMOOTHING_ALPHAS.get(mode.upper(), 0.50)
        self.state: Optional[Dict[str, Any]] = None

    def reset(self):
        """Resets tracking state (e.g. when changing videos or restarting)."""
        self.state = None

    def set_mode(self, mode: str):
        self.alpha = SMOOTHING_ALPHAS.get(mode.upper(), 0.50)

    def smooth_frame(
        self,
        pupil_center: Tuple[int, int],
        pupil_radius: float,
        iris_center: Tuple[int, int],
        iris_radius: float,
        iris_radius_x: float = 0.0,
        iris_radius_y: float = 0.0,
        iris_rotation: float = 0.0,
        is_detected: bool = True
    ) -> Dict[str, Any]:
        """
        Smooths current frame parameters using EMA with previous state.

        Returns dict with smoothed:
            pupil_center: (int, int)
            pupil_radius: float
            iris_center: (int, int)
            iris_radius: float
            iris_radius_x: float
            iris_radius_y: float
            iris_rotation: float
        """
        px, py = pupil_center
        pr = pupil_radius
        ix, iy = iris_center
        ir = iris_radius
        irx = iris_radius_x if iris_radius_x > 0 else ir
        iry = iris_radius_y if iris_radius_y > 0 else ir
        irot = iris_rotation

        if self.alpha >= 1.0 or self.state is None or not is_detected:
            # No smoothing or first frame
            smoothed_state = {
                "pupil_center": (int(round(px)), int(round(py))),
                "pupil_radius": float(pr),
                "iris_center": (int(round(ix)), int(round(iy))),
                "iris_radius": float(ir),
                "iris_radius_x": float(irx),
                "iris_radius_y": float(iry),
                "iris_rotation": float(irot)
            }
            if is_detected:
                self.state = smoothed_state
            return smoothed_state

        # Apply EMA: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        a = self.alpha
        b = 1.0 - a

        prev = self.state

        prev_px, prev_py = prev["pupil_center"]
        sm_px = a * px + b * prev_px
        sm_py = a * py + b * prev_py
        sm_pr = a * pr + b * prev["pupil_radius"]

        prev_ix, prev_iy = prev["iris_center"]
        sm_ix = a * ix + b * prev_ix
        sm_iy = a * iy + b * prev_iy
        sm_ir = a * ir + b * prev["iris_radius"]
        sm_irx = a * irx + b * prev["iris_radius_x"]
        sm_iry = a * iry + b * prev["iris_radius_y"]

        # Smooth rotation angle carefully (handle wraparound)
        prev_irot = prev["iris_rotation"]
        angle_diff = irot - prev_irot
        if angle_diff > 90:
            angle_diff -= 180
        elif angle_diff < -90:
            angle_diff += 180
        sm_irot = prev_irot + a * angle_diff

        smoothed_state = {
            "pupil_center": (int(round(sm_px)), int(round(sm_py))),
            "pupil_radius": float(sm_pr),
            "iris_center": (int(round(sm_ix)), int(round(sm_iy))),
            "iris_radius": float(sm_ir),
            "iris_radius_x": float(sm_irx),
            "iris_radius_y": float(sm_iry),
            "iris_rotation": float(sm_irot)
        }

        self.state = smoothed_state
        return smoothed_state
