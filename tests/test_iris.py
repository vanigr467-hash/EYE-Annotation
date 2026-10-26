"""Tests for the radial-ray + RANSAC ellipse iris detection pipeline."""

import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.synthetic_eye import generate_synthetic_eye
from src.preprocessing import preprocess_eye_image
from src.pupil_detection import detect_pupil
from src.iris_detection import detect_iris


class TestIrisDetection(unittest.TestCase):
    def setUp(self):
        self.img_rgb, self.gt = generate_synthetic_eye(
            width=600, height=450, pupil_r=45, iris_r=120
        )
        self.prep = preprocess_eye_image(self.img_rgb)
        self.pupil_res = detect_pupil(self.prep)

    def test_iris_detected(self):
        """Iris should be detected on a clean synthetic eye."""
        iris_res = detect_iris(self.prep, self.pupil_res)
        self.assertTrue(iris_res.detected, f"Iris should be detected. Message: {iris_res.message}")

    def test_iris_surrounds_pupil(self):
        """Iris radius must be larger than pupil radius."""
        iris_res = detect_iris(self.prep, self.pupil_res)
        if iris_res.detected:
            self.assertGreater(iris_res.radius, self.pupil_res.radius,
                               "Iris radius must be larger than pupil radius.")

    def test_iris_not_oversized(self):
        """Iris should not be much larger than ground truth (ratio check)."""
        iris_res = detect_iris(self.prep, self.pupil_res)
        if iris_res.detected:
            gt_r = self.gt["iris_radius"]
            # Allow up to 30% overshoot from ground truth
            self.assertLess(iris_res.radius, gt_r * 1.3,
                            f"Iris radius {iris_res.radius:.1f} too large vs GT {gt_r}")

    def test_iris_center_near_pupil(self):
        """Iris center should be near the pupil center."""
        iris_res = detect_iris(self.prep, self.pupil_res)
        if iris_res.detected:
            dx = iris_res.center[0] - self.pupil_res.center[0]
            dy = iris_res.center[1] - self.pupil_res.center[1]
            dist = np.sqrt(dx**2 + dy**2)
            self.assertLess(dist, self.pupil_res.radius * 0.8,
                            f"Iris center too far from pupil: {dist:.1f}px")

    def test_iris_has_inlier_points(self):
        """Detected iris should have inlier boundary evidence points."""
        iris_res = detect_iris(self.prep, self.pupil_res)
        if iris_res.detected:
            self.assertGreater(len(iris_res.inlier_points), 10,
                               "Should have at least 10 inlier boundary points.")

    def test_iris_quality_score(self):
        """Quality score should be positive for a detected iris."""
        iris_res = detect_iris(self.prep, self.pupil_res)
        if iris_res.detected:
            self.assertGreater(iris_res.quality_score, 20.0,
                               f"Quality score {iris_res.quality_score} should be > 20.")

    def test_ellipse_mode(self):
        """Ellipse fit mode should produce rx and ry values."""
        iris_res = detect_iris(self.prep, self.pupil_res, fit_mode="ellipse")
        if iris_res.detected:
            self.assertGreater(iris_res.radius_x, 0, "radius_x should be > 0 in ellipse mode.")
            self.assertGreater(iris_res.radius_y, 0, "radius_y should be > 0 in ellipse mode.")

    def test_no_pupil_gives_no_iris(self):
        """If pupil is not detected, iris should not be detected."""
        from src.pupil_detection import PupilResult
        fake_pupil = PupilResult(center=(0, 0), radius=0.0, detected=False)
        iris_res = detect_iris(self.prep, fake_pupil)
        self.assertFalse(iris_res.detected, "Iris should not be detected without a valid pupil.")


if __name__ == '__main__':
    unittest.main()
