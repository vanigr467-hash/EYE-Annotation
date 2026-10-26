import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.synthetic_eye import generate_synthetic_eye
from src.preprocessing import preprocess_eye_image
from src.pupil_detection import detect_pupil
from src.iris_detection import detect_iris
from src.annotation import generate_boundary_points, draw_automatic_annotations

class TestAnnotation(unittest.TestCase):
    def test_point_generation_counts(self):
        pts_32 = generate_boundary_points((300, 200), radius=50.0, num_points=32)
        pts_64 = generate_boundary_points((300, 200), radius=50.0, num_points=64)
        
        self.assertEqual(len(pts_32), 32)
        self.assertEqual(len(pts_64), 64)

    def test_annotation_drawing(self):
        img_rgb, _ = generate_synthetic_eye()
        prep_dict = preprocess_eye_image(img_rgb)
        p_res = detect_pupil(prep_dict)
        i_res = detect_iris(prep_dict, p_res)
        
        ann_img, pts_data = draw_automatic_annotations(img_rgb, p_res, i_res, num_points=64)
        self.assertEqual(ann_img.shape, img_rgb.shape)
        self.assertEqual(len(pts_data["pupil_points"]), 64)
        self.assertEqual(len(pts_data["iris_points"]), 64)

    def test_boundary_representation_styles(self):
        from src.annotation import draw_annotation_only_mask
        img_rgb, _ = generate_synthetic_eye()
        prep_dict = preprocess_eye_image(img_rgb)
        p_res = detect_pupil(prep_dict)
        i_res = detect_iris(prep_dict, p_res)

        styles = [
            "Circles / Ellipses",
            "Circles + Numbers",
            "Numbers Only",
            "Keypoints / Dots Only",
            "Full (Curves + Points + Numbers)"
        ]

        for st in styles:
            ann_img, _ = draw_automatic_annotations(
                img_rgb, p_res, i_res, num_points=32,
                boundary_style=st,
                show_values=True,
                values_format="Radius & Center (r, x, y)"
            )
            self.assertEqual(ann_img.shape, img_rgb.shape)

    def test_annotation_only_mask(self):
        from src.annotation import draw_annotation_only_mask
        img_rgb, _ = generate_synthetic_eye()
        prep_dict = preprocess_eye_image(img_rgb)
        p_res = detect_pupil(prep_dict)
        i_res = detect_iris(prep_dict, p_res)

        mask = draw_annotation_only_mask(
            img_rgb.shape, p_res, i_res, num_points=32,
            boundary_style="Circles + Numbers",
            show_values=True,
            values_format="Full Geometry (cx, cy, rx, ry)"
        )
        self.assertEqual(mask.shape, img_rgb.shape)
        # Verify the mask is on a black canvas with rendered annotation pixels
        self.assertGreater(np.sum(mask > 0), 100)
        # Corner of canvas should remain pure black (background)
        self.assertEqual(int(mask[0, 0, 0]), 0)

if __name__ == '__main__':
    unittest.main()
