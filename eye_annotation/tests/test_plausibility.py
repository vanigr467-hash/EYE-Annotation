"""
Unit tests for geometric plausibility validation.
Tests clean physiological eye geometry vs observed failure geometries.
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.plausibility import validate_detection
from src.pupil_detector import PupilResult
from src.iris_detector import IrisResult


class TestGeometricPlausibility(unittest.TestCase):
    def setUp(self):
        self.frame_shape = (480, 640, 3)

    def test_clean_physiological_detection_passes(self):
        """Clean concentric detection should pass all constraints."""
        pupil = {
            "cx": 320.0, "cy": 240.0,
            "rx": 28.0, "ry": 27.0,
            "radius": 27.5, "angle": 0.0,
            "detected": True
        }
        iris = {
            "cx": 322.0, "cy": 241.0,
            "rx": 70.0, "ry": 68.0,
            "radius": 69.0, "angle": 0.0,
            "detected": True
        }

        is_valid, failures, score = validate_detection(pupil, iris, self.frame_shape)
        self.assertTrue(is_valid)
        self.assertEqual(failures, [])
        self.assertGreaterEqual(score, 0.90)

    def test_observed_failure_case_rejected(self):
        """
        Exact observed failure from 01_eyes.mp4 screenshot:
        - Pupil locked on right edge eyelash/corner (cx=580, cy=240, r=80)
        - Iris radius giant (r=260, nearly frame height), centers far apart
        """
        bad_pupil = {
            "cx": 620.0,  # Right near frame border (W=640, margin ~28.8px)
            "cy": 250.0,
            "rx": 85.0,   # Way too large (0.18 * 480 = 86.4px)
            "ry": 80.0,
            "radius": 85.0,
            "angle": 0.0,
            "detected": True
        }
        bad_iris = {
            "cx": 450.0,  # Far away from pupil
            "cy": 220.0,
            "rx": 250.0,  # Spanning full frame height
            "ry": 240.0,
            "radius": 250.0,
            "angle": 0.0,
            "detected": True
        }

        is_valid, failures, score = validate_detection(bad_pupil, bad_iris, self.frame_shape)
        self.assertFalse(is_valid)
        self.assertIn("pupil_center_near_border", failures)
        self.assertIn("pupil_iris_centers_too_far", failures)
        self.assertIn("iris_radius_too_large", failures)
        self.assertLessEqual(score, 0.15)

    def test_pupil_outside_iris_rejected(self):
        """Pupil center outside the iris ellipse must fail."""
        pupil = {"cx": 200.0, "cy": 240.0, "radius": 25.0, "rx": 25.0, "ry": 25.0, "detected": True}
        iris = {"cx": 350.0, "cy": 240.0, "radius": 60.0, "rx": 60.0, "ry": 60.0, "detected": True}

        is_valid, failures, score = validate_detection(pupil, iris, self.frame_shape)
        self.assertFalse(is_valid)
        self.assertIn("pupil_center_outside_iris", failures)
        self.assertLessEqual(score, 0.15)

    def test_pupil_result_objects_compatibility(self):
        """Validates compatibility when PupilResult / IrisResult instances are passed."""
        p_res = PupilResult(center=(310, 230), radius=30.0, radius_x=30.0, radius_y=28.0, rotation=0.0, detected=True)
        i_res = IrisResult(center=(312, 231), radius=75.0, radius_x=75.0, radius_y=72.0, rotation=0.0, detected=True)

        is_valid, failures, score = validate_detection(p_res, i_res, self.frame_shape)
        self.assertTrue(is_valid)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
