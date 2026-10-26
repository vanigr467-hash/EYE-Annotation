"""Unit tests for robust ellipse fitting and geometric validation."""

import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ellipse_fitting import (
    ransac_ellipse_fit,
    validate_ellipse,
    point_to_ellipse_distance,
    generate_ellipse_points,
    refine_points_to_edge
)


class TestGeometry(unittest.TestCase):
    def test_ellipse_point_generation(self):
        cx, cy, rx, ry, ang = 200, 150, 60, 40, 15.0
        pts_64 = generate_ellipse_points(cx, cy, rx, ry, ang, num_points=64)
        self.assertEqual(len(pts_64), 64)
        
        pts_128 = generate_ellipse_points(cx, cy, rx, ry, ang, num_points=128)
        self.assertEqual(len(pts_128), 128)

    def test_ransac_ellipse_fit_clean(self):
        # Generate synthetic ellipse points
        cx, cy, rx, ry, ang = 300, 200, 80, 50, 0.0
        clean_pts = generate_ellipse_points(cx, cy, rx, ry, ang, num_points=60)
        
        # Add a few outliers
        pts_with_outliers = list(clean_pts) + [(10, 10), (500, 400), (300, 50)]
        pts_arr = np.array(pts_with_outliers, dtype=np.float32)

        fit_res = ransac_ellipse_fit(pts_arr, max_iterations=100, inlier_threshold=4.0)
        self.assertTrue(fit_res["success"], "RANSAC ellipse fit should succeed on clean ellipse")
        self.assertGreater(fit_res["inlier_ratio"], 0.75, "Inlier ratio should be high")

        (ecx, ecy), (ew, eh), _ = fit_res["ellipse"]
        erx, ery = ew / 2.0, eh / 2.0
        
        self.assertLess(abs(ecx - cx), 5.0, "Center X error should be < 5px")
        self.assertLess(abs(ecy - cy), 5.0, "Center Y error should be < 5px")
        self.assertLess(abs(max(erx, ery) - max(rx, ry)), 8.0, "Major axis error should be < 8px")

    def test_ellipse_validation(self):
        valid_ell = ((300.0, 200.0), (160.0, 120.0), 0.0)
        is_val, reason = validate_ellipse(valid_ell, {
            "min_radius": 20.0,
            "max_radius": 150.0,
            "max_eccentricity": 0.85,
            "image_shape": (400, 600)
        })
        self.assertTrue(is_val, f"Valid ellipse should pass validation: {reason}")

        # Test too small
        small_ell = ((300.0, 200.0), (10.0, 10.0), 0.0)
        is_val, reason = validate_ellipse(small_ell, {"min_radius": 20.0})
        self.assertFalse(is_val, "Too small ellipse should fail validation")


if __name__ == '__main__':
    unittest.main()
