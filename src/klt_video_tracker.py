"""
Pyramidal Lucas-Kanade (KLT) Optical Flow Feature Tracker and Kalman Fusion Module.
Provides sub-pixel jitter-free continuous video tracking across fast eye movements,
glints, and partial eyelid occlusions.
"""

from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
from src.tracker import EyeKalmanFilter
from src.ellipse_fitting import generate_ellipse_points, fit_ellipse_fitzgibbon


class KLTVideoTracker:
    """
    Sub-pixel Video Eye Tracker fusing Pyramidal Lucas-Kanade Optical Flow 
    with 10D Linear Kalman Filter state estimation.
    """

    def __init__(self, fps: float = 25.0):
        self.kf = EyeKalmanFilter(dt=1.0 / max(1.0, fps))
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pupil_pts: Optional[np.ndarray] = None
        self.prev_iris_pts: Optional[np.ndarray] = None
        self.last_valid_state: Optional[Dict[str, Any]] = None
        self.tracking_status = "INITIALIZING"
        self.frame_index = 0
        self.consecutive_failures = 0

    def reset(self):
        self.kf = EyeKalmanFilter()
        self.prev_gray = None
        self.prev_pupil_pts = None
        self.prev_iris_pts = None
        self.last_valid_state = None
        self.tracking_status = "INITIALIZING"
        self.frame_index = 0
        self.consecutive_failures = 0

    def init_from_detection(
        self,
        gray_img: np.ndarray,
        pupil_res: Any,
        iris_res: Any
    ) -> Dict[str, Any]:
        """
        Initializes KLT boundary feature points and Kalman filter from a verified detection.
        """
        px, py = pupil_res.center
        pr = pupil_res.radius
        prx = getattr(pupil_res, "radius_x", pr)
        pry = getattr(pupil_res, "radius_y", pr)
        prot = getattr(pupil_res, "rotation", 0.0)

        ix, iy = iris_res.center
        ir = iris_res.radius
        irx = getattr(iris_res, "radius_x", ir)
        iry = getattr(iris_res, "radius_y", ir)
        irot = getattr(iris_res, "rotation", 0.0)

        self.kf.init_state(px, py, pr, ix, iy, ir)

        # Generate 32 sub-pixel perimeter feature points
        p_pts = generate_ellipse_points(px, py, prx, pry, prot, num_points=32)
        i_pts = generate_ellipse_points(ix, iy, irx, iry, irot, num_points=32)

        self.prev_gray = gray_img.copy()
        self.prev_pupil_pts = np.array(p_pts, dtype=np.float32).reshape(-1, 1, 2)
        self.prev_iris_pts = np.array(i_pts, dtype=np.float32).reshape(-1, 1, 2)

        self.tracking_status = "TRACKING"
        self.consecutive_failures = 0

        current_state = {
            "detection_status": "DETECTED",
            "tracking_status": "LOCKED",
            "pupil_center": (int(round(px)), int(round(py))),
            "pupil_radius": float(pr),
            "pupil_radius_x": float(prx),
            "pupil_radius_y": float(pry),
            "pupil_rotation": float(prot),
            "iris_center": (int(round(ix)), int(round(iy))),
            "iris_radius": float(ir),
            "iris_radius_x": float(irx),
            "iris_radius_y": float(iry),
            "iris_rotation": float(irot),
            "pupil_confidence": pupil_res.confidence,
            "iris_confidence": iris_res.confidence,
            "overall_confidence": round(pupil_res.confidence * 0.55 + iris_res.confidence * 0.45, 1)
        }
        self.last_valid_state = current_state
        return current_state

    def track_frame_klt(
        self,
        curr_gray: np.ndarray,
        detector_pupil_res: Optional[Any] = None,
        detector_iris_res: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes Lucas-Kanade optical flow on boundary points with Forward-Backward
        error filtering and Kalman state fusion.
        """
        self.frame_index += 1

        # If we have a fresh, verified high-confidence detector result, re-anchor
        if (detector_pupil_res and detector_iris_res and 
            getattr(detector_pupil_res, "detected", False) and 
            getattr(detector_iris_res, "detected", False) and
            detector_pupil_res.confidence >= 70.0 and detector_iris_res.confidence >= 65.0):
            return self.init_from_detection(curr_gray, detector_pupil_res, detector_iris_res)

        # Fallback if no prior tracking state
        if self.prev_gray is None or self.prev_pupil_pts is None or self.last_valid_state is None:
            if detector_pupil_res and getattr(detector_pupil_res, "detected", False):
                return self.init_from_detection(curr_gray, detector_pupil_res, detector_iris_res)
            else:
                self.tracking_status = "LOST"
                return {
                    "detection_status": "NOT_DETECTED",
                    "tracking_status": "LOST",
                    "pupil_center": (0, 0), "pupil_radius": 0.0,
                    "iris_center": (0, 0), "iris_radius": 0.0,
                    "pupil_confidence": 0.0, "iris_confidence": 0.0, "overall_confidence": 0.0
                }

        # --- Lucas-Kanade Optical Flow with Forward-Backward Consistency ---
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        try:
            # 1. Track Pupil Points Forward & Backward
            p_next, status_p, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pupil_pts, None, **lk_params)
            if p_next is not None and status_p is not None:
                p_back, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self.prev_gray, p_next, None, **lk_params)
                if p_back is not None:
                    fb_err_p = np.linalg.norm(self.prev_pupil_pts - p_back, axis=2).ravel()
                    valid_p_mask = (status_p.ravel() == 1) & (fb_err_p < 2.0)
                else:
                    valid_p_mask = (status_p.ravel() == 1)
            else:
                valid_p_mask = np.zeros(len(self.prev_pupil_pts), dtype=bool)

            # 2. Track Iris Points Forward & Backward
            i_next, status_i, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_iris_pts, None, **lk_params)
            if i_next is not None and status_i is not None:
                i_back, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self.prev_gray, i_next, None, **lk_params)
                if i_back is not None:
                    fb_err_i = np.linalg.norm(self.prev_iris_pts - i_back, axis=2).ravel()
                    valid_i_mask = (status_i.ravel() == 1) & (fb_err_i < 2.0)
                else:
                    valid_i_mask = (status_i.ravel() == 1)
            else:
                valid_i_mask = np.zeros(len(self.prev_iris_pts), dtype=bool)

            valid_p_count = int(np.sum(valid_p_mask))
            valid_i_count = int(np.sum(valid_i_mask))

            if valid_p_count >= 10:
                # Fit new pupil ellipse on tracked optical flow points
                tracked_p_pts = p_next[valid_p_mask].reshape(-1, 2)
                ell_p = fit_ellipse_fitzgibbon(tracked_p_pts)
                if ell_p is not None:
                    (pcx, pcy), (pw, ph), pang = ell_p
                    prx, pry = pw / 2.0, ph / 2.0
                    pr = (prx + pry) / 2.0
                else:
                    pcx = float(np.mean(tracked_p_pts[:, 0]))
                    pcy = float(np.mean(tracked_p_pts[:, 1]))
                    pr = self.last_valid_state["pupil_radius"]
                    prx, pry, pang = pr, pr, 0.0
            else:
                # Use Kalman state prediction
                pred = self.kf.predict()
                pcx, pcy = pred.get("pupil_center", self.last_valid_state["pupil_center"])
                pr = pred.get("pupil_radius", self.last_valid_state["pupil_radius"])
                prx = self.last_valid_state.get("pupil_radius_x", pr)
                pry = self.last_valid_state.get("pupil_radius_y", pr)
                pang = self.last_valid_state.get("pupil_rotation", 0.0)

            if valid_i_count >= 10:
                # Fit new iris ellipse on tracked optical flow points
                tracked_i_pts = i_next[valid_i_mask].reshape(-1, 2)
                ell_i = fit_ellipse_fitzgibbon(tracked_i_pts)
                if ell_i is not None:
                    (icx, icy), (iw, ih), iang = ell_i
                    irx, iry = iw / 2.0, ih / 2.0
                    ir = (irx + iry) / 2.0
                else:
                    icx = float(np.mean(tracked_i_pts[:, 0]))
                    icy = float(np.mean(tracked_i_pts[:, 1]))
                    ir = self.last_valid_state["iris_radius"]
                    irx, iry, iang = ir, ir, 0.0
            else:
                pred = self.kf.predict()
                icx, icy = pred.get("iris_center", self.last_valid_state["iris_center"])
                ir = pred.get("iris_radius", self.last_valid_state["iris_radius"])
                irx = self.last_valid_state.get("iris_radius_x", ir)
                iry = self.last_valid_state.get("iris_radius_y", ir)
                iang = self.last_valid_state.get("iris_rotation", 0.0)

            # Update Kalman state
            self.kf.predict()
            self.kf.update(pcx, pcy, pr, icx, icy, ir)

            # Re-generate smooth perimeter points for next frame
            new_p_pts = generate_ellipse_points(pcx, pcy, prx, pry, pang, num_points=32)
            new_i_pts = generate_ellipse_points(icx, icy, irx, iry, iang, num_points=32)

            self.prev_gray = curr_gray.copy()
            self.prev_pupil_pts = np.array(new_p_pts, dtype=np.float32).reshape(-1, 1, 2)
            self.prev_iris_pts = np.array(new_i_pts, dtype=np.float32).reshape(-1, 1, 2)

            p_flow_ratio = valid_p_count / 32.0
            i_flow_ratio = valid_i_count / 32.0

            p_conf = float(np.clip(p_flow_ratio * 70.0 + 25.0, 30.0, 96.0))
            i_conf = float(np.clip(i_flow_ratio * 70.0 + 25.0, 30.0, 94.0))

            current_state = {
                "detection_status": "DETECTED" if p_flow_ratio >= 0.40 else "PREDICTED",
                "tracking_status": "OPTICAL_FLOW_LOCKED" if p_flow_ratio >= 0.50 else "PREDICTING",
                "pupil_center": (int(round(pcx)), int(round(pcy))),
                "pupil_radius": float(pr),
                "pupil_radius_x": float(prx),
                "pupil_radius_y": float(pry),
                "pupil_rotation": float(pang),
                "iris_center": (int(round(icx)), int(round(icy))),
                "iris_radius": float(ir),
                "iris_radius_x": float(irx),
                "iris_radius_y": float(iry),
                "iris_rotation": float(iang),
                "pupil_confidence": p_conf,
                "iris_confidence": i_conf,
                "overall_confidence": round(p_conf * 0.55 + i_conf * 0.45, 1)
            }
            self.last_valid_state = current_state
            return current_state

        except Exception:
            # Fallback
            pred = self.kf.predict()
            return self.last_valid_state
