import unittest
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.synthetic_eye import generate_synthetic_eye_video
from src.video_processor import process_eye_video

class TestVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "scratch_test_video.avi"))
        cls.output_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "scratch_out_video.avi"))
        generate_synthetic_eye_video(cls.test_video_path, total_frames=15, fps=25.0)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video_path):
            try: os.remove(cls.test_video_path)
            except Exception: pass
        if os.path.exists(cls.output_video_path):
            try: os.remove(cls.output_video_path)
            except Exception: pass

    def test_video_processing(self):
        results = process_eye_video(
            self.test_video_path,
            self.output_video_path,
            detection_mode="FAST",
            frame_skip=1,
            num_points=32
        )
        self.assertTrue(os.path.exists(results["output_video_path"]), "Output video file should exist")
        self.assertGreaterEqual(results["processed_frames"], 10)
        self.assertGreater(results["pupil_detection_rate"], 70.0)
        self.assertEqual(len(results["frame_logs"]), results["processed_frames"])

if __name__ == '__main__':
    unittest.main()
