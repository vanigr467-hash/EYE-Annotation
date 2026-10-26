"""
Kalman Filter-based Temporal Video Eye Tracker.

Maintains continuous eye tracking across video frames with:
- Constant Velocity Kalman State Estimation [px, py, pr, ix, iy, ir, v_px, v_py, v_ix, v_iy]
- Dynamic ROI Prediction for accelerated localized search
- Detection Failure Recovery & Graceful Prediction Decay
- Adaptive Periodic Full Re-detection Triggers
- Temporal Jump & Outlier Rejection
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class EyeKalmanFilter:
    """
    Linear Kalman Filter estimating 2D position, radius, and velocities
    for both Pupil and Iris.
    State: [px, py, pr, ix, iy, ir, vpx, vpy, vix, viy]^T (10-dim)
    """

    def __init__(self, dt: float = 1.0 / 25.0):
        self.dim_x = 10
        self.dim_z = 6  # Measurement: [px, py, pr, ix, iy, ir]
        self.dt = dt

        # State transition matrix F (constant velocity motion model)
        self.F = np.eye(self.dim_x, dtype=np.float64)
        self.F[0, 6] = dt  # px += vpx * dt
        self.F[1, 7] = dt  # py += vpy * dt
        self.F[3, 8] = dt  # ix += vix * dt
        self.F[4, 9] = dt  # iy += viy * dt

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        for i in range(self.dim_z):
            self.H[i, i] = 1.0

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x, dtype=np.float64) * 0.5
        self.Q[6:, 6:] *= 2.0  # higher velocity uncertainty

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z, dtype=np.float64) * 2.0

        # State covariance P
        self.P = np.eye(self.dim_x, dtype=np.float64) * 10.0

        # State vector x
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.initialized = False

    def init_state(self, px: float, py: float, pr: float, ix: float, iy: float, ir: float):
        self.x = np.array([[px], [py], [pr], [ix], [iy], [ir], [0], [0], [0], [0]], dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64) * 5.0
        self.initialized = True

    def predict(self) -> Dict[str, Any]:
        if not self.initialized:
            return {}
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return {
            "pupil_center": (float(self.x[0, 0]), float(self.x[1, 0])),
            "pupil_radius": float(self.x[2, 0]),
            "iris_center": (float(self.x[3, 0]), float(self.x[4, 0])),
            "iris_radius": float(self.x[5, 0]),
            "pupil_velocity": (float(self.x[6, 0]), float(self.x[7, 0])),
            "iris_velocity": (float(self.x[8, 0]), float(self.x[9, 0]))
        }

    def update(self, px: float, py: float, pr: float, ix: float, iy: float, ir: float):
        if not self.initialized:
            self.init_state(px, py, pr, ix, iy, ir)
            return

        z = np.array([[px], [py], [pr], [ix], [iy], [ir]], dtype=np.float64)
        y = z - (self.H @ self.x)  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P


from src.config_constants import (
    MAX_TRACKING_JUMP_RATIO,
    MAX_CONSECUTIVE_GATED_FRAMES,
    BLINK_INTENSITY_DROP_THRESH
)


class EyeTracker:
    """
    Manages video temporal eye tracking, failure recovery, smoothing, 
    gating, and loss handling.
    """

    def __init__(
        self,
        smoothing_mode: str = "MEDIUM",
        max_consecutive_failures: int = MAX_CONSECUTIVE_GATED_FRAMES,
        fps: float = 25.0
    ):
        self.kf = EyeKalmanFilter(dt=1.0 / max(1.0, fps))
        self.max_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.consecutive_gated = 0
        self.last_valid_state: Optional[Dict[str, Any]] = None
        self.frame_index = 0
        self.tracking_status = "INITIALIZING"

    def reset(self):
        self.kf = EyeKalmanFilter()
        self.consecutive_failures = 0
        self.consecutive_gated = 0
        self.last_valid_state = None
        self.frame_index = 0
        self.tracking_status = "INITIALIZING"

    def get_roi_bounding_box(
        self,
        image_shape: Tuple[int, int],
        margin_factor: float = 2.2
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Computes predicted search ROI bounding box (ymin, xmin, ymax, xmax)
        around the predicted eye center.
        """
        if self.last_valid_state is None:
            return None

        h, w = image_shape[:2]
        ix, iy = self.last_valid_state["iris_center"]
        ir = self.last_valid_state["iris_radius"]

        roi_half = int(max(30.0, ir * margin_factor))
        xmin = max(0, int(ix - roi_half))
        xmax = min(w, int(ix + roi_half))
        ymin = max(0, int(iy - roi_half))
        ymax = min(h, int(iy + roi_half))

        if (xmax - xmin) < 30 or (ymax - ymin) < 30:
            return None

        return (ymin, xmin, ymax, xmax)

    def process_frame_tracking(
        self,
        pupil_res: Any,
        iris_res: Any,
        val_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates tracking state, executes Kalman predict/update, and handles recovery.
        """
        self.frame_index += 1

        is_valid = val_result.get("is_valid", False) and getattr(pupil_res, "detected", False)
        iris_valid = val_result.get("iris_valid", False) and getattr(iris_res, "detected", False)

        # 0. Blink Detection check
        if val_result.get("is_blink", False):
            self.tracking_status = "BLINK"
            return {
                "detection_status": "BLINK",
                "tracking_status": "BLINK",
                "pupil_center": (0, 0),
                "pupil_radius": 0.0,
                "iris_center": (0, 0),
                "iris_radius": 0.0,
                "iris_radius_x": 0.0,
                "iris_radius_y": 0.0,
                "iris_rotation": 0.0,
                "pupil_confidence": 0.0,
                "iris_confidence": 0.0,
                "overall_confidence": 0.0,
                "reason": "blink"
            }

        if is_valid:
            px, py = pupil_res.center
            pr = pupil_res.radius

            if iris_valid:
                ix, iy = iris_res.center
                ir = iris_res.radius
                irx = getattr(iris_res, "radius_x", ir)
                iry = getattr(iris_res, "radius_y", ir)
                irot = getattr(iris_res, "rotation", 0.0)
                i_conf = iris_res.confidence
            else:
                ix, iy = px, py
                ir = pr * 2.5
                irx, iry, irot = ir, ir, 0.0
                i_conf = 25.0

            # 1. Kalman Gating Check (reject trajectory jumps > 0.5 * r_iris)
            is_gated = False
            if self.kf.initialized:
                pred = self.kf.predict()
                pred_px, pred_py = pred.get("pupil_center", (px, py))
                jump = np.hypot(px - pred_px, py - pred_py)
                jump_threshold = max(20.0, ir * MAX_TRACKING_JUMP_RATIO)
                if jump > jump_threshold:
                    is_gated = True

            if is_gated:
                self.consecutive_gated += 1
                if self.consecutive_gated >= self.max_failures:
                    self.tracking_status = "LOST"
                    return {
                        "detection_status": "LOST",
                        "tracking_status": "LOST",
                        "pupil_center": (0, 0),
                        "pupil_radius": 0.0,
                        "iris_center": (0, 0),
                        "iris_radius": 0.0,
                        "iris_radius_x": 0.0,
                        "iris_radius_y": 0.0,
                        "iris_rotation": 0.0,
                        "pupil_confidence": 0.0,
                        "iris_confidence": 0.0,
                        "overall_confidence": 0.0,
                        "reason": "trajectory_jump_lost",
                        "review_required": True
                    }
                else:
                    self.tracking_status = "GATED"
                    pred = self.kf.predict()
                    pred_px, pred_py = pred.get("pupil_center", (px, py))
                    pred_ix, pred_iy = pred.get("iris_center", (ix, iy))
                    decay = 0.65 ** self.consecutive_gated
                    return {
                        "detection_status": "GATED",
                        "tracking_status": "GATED",
                        "pupil_center": (int(round(pred_px)), int(round(pred_py))),
                        "pupil_radius": round(float(self.kf.x[2, 0]), 1),
                        "iris_center": (int(round(pred_ix)), int(round(pred_iy))),
                        "iris_radius": round(float(self.kf.x[5, 0]), 1),
                        "iris_radius_x": round(irx, 1),
                        "iris_radius_y": round(iry, 1),
                        "iris_rotation": round(irot, 1),
                        "pupil_confidence": round(15.0 * decay, 1),
                        "iris_confidence": round(15.0 * decay, 1),
                        "overall_confidence": round(15.0 * decay, 1),
                        "reason": "kalman_jump_gated"
                    }

            # Successful Frame Detection (Ungated)
            self.consecutive_failures = 0
            self.consecutive_gated = 0
            self.tracking_status = "TRACKING"

            # Kalman predict & update
            if not self.kf.initialized:
                self.kf.init_state(px, py, pr, ix, iy, ir)
            else:
                self.kf.predict()
                self.kf.update(px, py, pr, ix, iy, ir)

            smoothed_px = int(round(self.kf.x[0, 0]))
            smoothed_py = int(round(self.kf.x[1, 0]))
            smoothed_pr = float(self.kf.x[2, 0])
            smoothed_ix = int(round(self.kf.x[3, 0]))
            smoothed_iy = int(round(self.kf.x[4, 0]))
            smoothed_ir = float(self.kf.x[5, 0])

            status_str = "TRACKED" if self.frame_index > 1 else "DETECTED"

            current_state = {
                "detection_status": status_str,
                "tracking_status": "ACTIVE",
                "pupil_center": (smoothed_px, smoothed_py),
                "pupil_radius": round(smoothed_pr, 1),
                "iris_center": (smoothed_ix, smoothed_iy),
                "iris_radius": round(smoothed_ir, 1),
                "iris_radius_x": round(irx, 1),
                "iris_radius_y": round(iry, 1),
                "iris_rotation": round(irot, 1),
                "pupil_confidence": pupil_res.confidence,
                "iris_confidence": i_conf,
                "overall_confidence": round(pupil_res.confidence * 0.55 + i_conf * 0.45, 1)
            }
            self.last_valid_state = current_state
            return current_state

        else:
            # Detection Failed on this frame -> Use Kalman Prediction
            self.consecutive_failures += 1

            if self.last_valid_state is not None and self.consecutive_failures <= self.max_failures:
                self.tracking_status = "PREDICTED"
                pred = self.kf.predict() if self.kf.initialized else {}
                decay = 0.82 ** self.consecutive_failures

                px = int(round(pred.get("pupil_center", self.last_valid_state["pupil_center"])[0]))
                py = int(round(pred.get("pupil_center", self.last_valid_state["pupil_center"])[1]))
                ix = int(round(pred.get("iris_center", self.last_valid_state["iris_center"])[0]))
                iy = int(round(pred.get("iris_center", self.last_valid_state["iris_center"])[1]))

                pred_state = {
                    "detection_status": "PREDICTED",
                    "tracking_status": "PREDICTING",
                    "pupil_center": (px, py),
                    "pupil_radius": self.last_valid_state["pupil_radius"],
                    "iris_center": (ix, iy),
                    "iris_radius": self.last_valid_state["iris_radius"],
                    "iris_radius_x": self.last_valid_state.get("iris_radius_x", self.last_valid_state["iris_radius"]),
                    "iris_radius_y": self.last_valid_state.get("iris_radius_y", self.last_valid_state["iris_radius"]),
                    "iris_rotation": self.last_valid_state.get("iris_rotation", 0.0),
                    "pupil_confidence": round(max(10.0, self.last_valid_state["pupil_confidence"] * decay), 1),
                    "iris_confidence": round(max(10.0, self.last_valid_state["iris_confidence"] * decay), 1),
                    "overall_confidence": round(max(10.0, self.last_valid_state["overall_confidence"] * decay), 1)
                }
                return pred_state
            else:
                self.tracking_status = "FAILED"
                return {
                    "detection_status": "NOT_DETECTED",
                    "tracking_status": "LOST",
                    "pupil_center": (0, 0),
                    "pupil_radius": 0.0,
                    "iris_center": (0, 0),
                    "iris_radius": 0.0,
                    "iris_radius_x": 0.0,
                    "iris_radius_y": 0.0,
                    "iris_rotation": 0.0,
                    "pupil_confidence": 0.0,
                    "iris_confidence": 0.0,
                    "overall_confidence": 0.0
                }
