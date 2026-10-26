"""Unit tests for Kalman temporal eye tracking and prediction."""

import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tracker import EyeTracker, EyeKalmanFilter
from src.pupil_detector import PupilResult
from src.iris_detector import IrisResult


class TestTracking(unittest.TestCase):
    def test_kalman_initialization_and_prediction(self):
        kf = EyeKalmanFilter(dt=1.0/25.0)
        kf.init_state(px=300, py=200, pr=40, ix=302, iy=201, ir=110)
        
        pred = kf.predict()
        self.assertIn("pupil_center", pred)
        self.assertAlmostEqual(pred["pupil_center"][0], 300.0, delta=2.0)
        self.assertAlmostEqual(pred["pupil_center"][1], 200.0, delta=2.0)

    def test_tracker_temporal_smoothing(self):
        tracker = EyeTracker(fps=25.0)

        # Frame 1: Valid detection
        p1 = PupilResult(center=(300, 200), radius=45.0, confidence=90.0, detected=True)
        i1 = IrisResult(center=(300, 200), radius=120.0, confidence=85.0, detected=True)
        val1 = {"is_valid": True, "pupil_valid": True, "iris_valid": True}

        res1 = tracker.process_frame_tracking(p1, i1, val1)
        self.assertEqual(res1["detection_status"], "DETECTED")

        # Frame 2: Missing frame (failure recovery via prediction)
        p_fail = PupilResult(center=(0, 0), radius=0.0, detected=False)
        i_fail = IrisResult(center=(0, 0), radius=0.0, detected=False)
        val_fail = {"is_valid": False}

        res2 = tracker.process_frame_tracking(p_fail, i_fail, val_fail)
        self.assertEqual(res2["detection_status"], "PREDICTED")
        self.assertGreater(res2["pupil_center"][0], 250)


if __name__ == '__main__':
    unittest.main()
