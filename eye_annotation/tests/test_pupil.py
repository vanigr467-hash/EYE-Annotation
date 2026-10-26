import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.synthetic_eye import generate_synthetic_eye
from src.preprocessing import preprocess_eye_image
from src.pupil_detection import detect_pupil

class TestPupilDetection(unittest.TestCase):
    def setUp(self):
        self.img_rgb, self.gt = generate_synthetic_eye(width=600, height=450, pupil_r=45, iris_r=120)
        self.prep_dict = preprocess_eye_image(self.img_rgb)

    def test_pupil_detection_accuracy(self):
        pupil_res = detect_pupil(self.prep_dict, sensitivity=0.5)
        self.assertTrue(pupil_res.detected, "Pupil should be detected")
        
        gt_cx, gt_cy = self.gt["pupil_center"]
        gt_r = self.gt["pupil_radius"]

        det_cx, det_cy = pupil_res.center
        det_r = pupil_res.radius

        dist_error = np.sqrt((gt_cx - det_cx)**2 + (gt_cy - det_cy)**2)
        radius_error = abs(gt_r - det_r)

        self.assertLess(dist_error, 10.0, f"Pupil center error {dist_error:.2f}px is too large")
        self.assertLess(radius_error, 8.0, f"Pupil radius error {radius_error:.2f}px is too large")
        self.assertGreater(pupil_res.confidence, 50.0, "Pupil confidence should be > 50%")

if __name__ == '__main__':
    unittest.main()
