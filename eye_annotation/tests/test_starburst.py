import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.synthetic_eye import generate_synthetic_eye
from src.preprocessing import preprocess_eye_image
from src.starburst_refinement import refine_pupil_boundary_starburst

class TestStarburstRefinement(unittest.TestCase):
    def setUp(self):
        self.img_rgb, self.gt = generate_synthetic_eye(width=600, height=450, pupil_r=45, iris_r=120)
        self.prep = preprocess_eye_image(self.img_rgb)

    def test_starburst_ray_tracing(self):
        gt_center = self.gt["pupil_center"]
        gt_r = self.gt["pupil_radius"]

        # Provide slightly noisy initial center
        noisy_center = (gt_center[0] + 3, gt_center[1] - 2)

        res = refine_pupil_boundary_starburst(
            self.prep["gray_inpainted"],
            noisy_center,
            gt_r,
            num_rays=48
        )

        ref_cx, ref_cy = res["refined_center"]
        ref_r = res["refined_radius"]

        center_err = np.sqrt((gt_center[0] - ref_cx)**2 + (gt_center[1] - ref_cy)**2)
        radius_err = abs(gt_r - ref_r)

        self.assertLess(center_err, 5.0, f"Starburst refined center error {center_err:.2f}px should be < 5px")
        self.assertLess(radius_err, 5.0, f"Starburst refined radius error {radius_err:.2f}px should be < 5px")
        self.assertGreater(len(res["inlier_points"]), 20, "Should extract at least 20 inlier points")

if __name__ == '__main__':
    unittest.main()
