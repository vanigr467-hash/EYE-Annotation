import unittest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gaze_tracking import calculate_gaze_vector

class TestGazeTracking(unittest.TestCase):
    def test_gaze_calculation_center(self):
        g_info = calculate_gaze_vector((300, 200), (300, 200), iris_radius=100.0)
        self.assertEqual(g_info["direction_label"], "Direct / Center")
        self.assertEqual(g_info["distance_px"], 0.0)

    def test_gaze_calculation_right(self):
        g_info = calculate_gaze_vector((325, 200), (300, 200), iris_radius=100.0)
        self.assertEqual(g_info["direction_label"], "Looking Right")
        self.assertEqual(g_info["gaze_angle_deg"], 0.0)

    def test_gaze_calculation_down_left(self):
        g_info = calculate_gaze_vector((280, 230), (300, 200), iris_radius=100.0)
        self.assertIn("Left", g_info["direction_label"])

if __name__ == '__main__':
    unittest.main()
