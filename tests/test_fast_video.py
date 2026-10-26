"""
Unit and benchmark tests for the ultra-fast video annotator engine.
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import numpy as np
import cv2
from src.fast_video_annotator import _fast_detect_anchor, process_eye_video_instant
from utils.synthetic_eye import generate_synthetic_eye

class TestFastVideoAnnotator(unittest.TestCase):
    def test_fast_anchor_detection(self):
        img_rgb, gt = generate_synthetic_eye(pupil_offset=(4, -2))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        pupil, iris = _fast_detect_anchor(img_bgr)
        self.assertTrue(pupil["detected"])
        self.assertTrue(iris["detected"])

        gt_px, gt_py = gt["pupil_center"]
        self.assertAlmostEqual(pupil["cx"], gt_px, delta=50.0)
        self.assertAlmostEqual(pupil["cy"], gt_py, delta=50.0)

        # Iris radius must surround pupil
        self.assertGreater(iris["rx"], pupil["rx"] * 1.3)

    def test_instant_video_processing(self):
        sample_path = "sample_videos/01_eyes.mp4"
        if not os.path.exists(sample_path):
            self.skipTest("Sample video 01_eyes.mp4 not found")

        output_path = "outputs/test_instant_run.mp4"
        res = process_eye_video_instant(sample_path, output_path)

        self.assertIn("output_video_path", res)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 1000)

        # High-speed verification (> 10 FPS overall on CPU including full anchor detection)
        self.assertGreater(res["processing_fps"], 10.0)
        self.assertLess(res["processing_time_sec"], 15.0)
        self.assertEqual(res["processed_frames"], 90)

    def test_optical_flow_none_status_robustness(self):
        """Verify that process_eye_video_instant handles None from calcOpticalFlowPyrLK without AttributeError."""
        from unittest.mock import patch
        sample_path = "sample_videos/01_eyes.mp4"
        if not os.path.exists(sample_path):
            self.skipTest("Sample video 01_eyes.mp4 not found")

        output_path = "outputs/test_none_flow.mp4"
        # Simulate calcOpticalFlowPyrLK returning (None, None, None)
        with patch("cv2.calcOpticalFlowPyrLK", return_value=(None, None, None)):
            res = process_eye_video_instant(sample_path, output_path)
            self.assertIn("output_video_path", res)
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(res["processed_frames"], 90)

    def test_render_mode_both_and_annotation_only(self):
        """Verify dual outputs and annotation-only mask video outputs."""
        sample_path = "sample_videos/01_eyes.mp4"
        if not os.path.exists(sample_path):
            self.skipTest("Sample video 01_eyes.mp4 not found")

        output_both = "outputs/test_both.mp4"
        res = process_eye_video_instant(
            sample_path,
            output_both,
            frame_skip=5,
            render_mode="both",
            annotation_style="Circles + Numbers",
            show_values=True,
            values_format="Radius & Center (r, x, y)"
        )

        self.assertIsNotNone(res["annotated_video_path"])
        self.assertIsNotNone(res["annotation_only_video_path"])
        self.assertTrue(os.path.exists(res["annotated_video_path"]))
        self.assertTrue(os.path.exists(res["annotation_only_video_path"]))
        self.assertGreater(len(res["annotation_only_frames"]), 0)

if __name__ == "__main__":
    unittest.main()
