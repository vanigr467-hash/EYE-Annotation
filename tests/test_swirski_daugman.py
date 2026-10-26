"""Unit tests for Swirski pupil localization, Daugman iris, and KLT tracker."""

import unittest
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.synthetic_eye import generate_synthetic_eye
from src.preprocessing import preprocess_eye_image
from src.swirski_pupil import detect_pupil_swirski, fit_ellipse_fitzgibbon
from src.daugman_iris import detect_iris_daugman
from src.klt_video_tracker import KLTVideoTracker
from src.pupil_detector import PupilResult


class TestSwirskiDaugmanKLT(unittest.TestCase):
    def setUp(self):
        self.img_rgb, self.gt = generate_synthetic_eye(
            width=600, height=450, pupil_r=45, iris_r=120
        )
        self.prep = preprocess_eye_image(self.img_rgb)

    def test_swirski_pupil_detection(self):
        """Swirski pipeline should detect pupil accurately on synthetic eye."""
        res = detect_pupil_swirski(self.prep["gray_inpainted"])
        self.assertTrue(res["detected"])
        cx, cy = res["center"]
        gt_cx, gt_cy = self.gt["pupil_center"]
        self.assertLess(abs(cx - gt_cx), 10.0, f"Pupil cx error too large: {abs(cx - gt_cx)}")
        self.assertLess(abs(cy - gt_cy), 10.0, f"Pupil cy error too large: {abs(cy - gt_cy)}")
        self.assertLess(abs(res["radius"] - self.gt["pupil_radius"]), 10.0)

    def test_daugman_iris_detection(self):
        """Daugman integro-differential should accurately detect iris boundary."""
        p_res = PupilResult(
            center=self.gt["pupil_center"],
            radius=float(self.gt["pupil_radius"]),
            detected=True
        )
        res = detect_iris_daugman(self.prep["gray_inpainted"], p_res)
        self.assertTrue(res["detected"])
        self.assertLess(abs(res["radius"] - self.gt["iris_radius"]), 15.0)

    def test_klt_tracker_lifecycle(self):
        """KLT tracker should initialize and track consecutive frames."""
        tracker = KLTVideoTracker(fps=25.0)
        p_res = PupilResult(
            center=self.gt["pupil_center"],
            radius=float(self.gt["pupil_radius"]),
            confidence=95.0,
            detected=True
        )
        from src.iris_detector import IrisResult
        i_res = IrisResult(
            center=self.gt["iris_center"],
            radius=float(self.gt["iris_radius"]),
            confidence=90.0,
            detected=True
        )
        gray1 = self.prep["gray_inpainted"]
        init_state = tracker.init_from_detection(gray1, p_res, i_res)
        self.assertEqual(init_state["tracking_status"], "LOCKED")

        gray2 = np.roll(np.roll(gray1, 2, axis=0), 3, axis=1)
        track_state = tracker.track_frame_klt(gray2)
        self.assertTrue(track_state["detection_status"] in ["DETECTED", "PREDICTED"])
        self.assertGreater(track_state["overall_confidence"], 50.0)


if __name__ == '__main__':
    unittest.main()
