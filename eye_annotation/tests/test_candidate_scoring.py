"""
Unit tests for Pupil Candidate Scoring and Confidence Calibration.
Validates 5-factor scoring, plausibility gating, and contrast rejection of eyelashes.
"""

import unittest
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.synthetic_eye import generate_synthetic_eye
from src.candidate_scoring import score_pupil_candidate
from src.confidence import compute_calibrated_confidence


class TestCandidateScoring(unittest.TestCase):
    def setUp(self):
        self.img_rgb, self.gt = generate_synthetic_eye(width=600, height=450, pupil_r=40, iris_r=110)
        self.gray = np.mean(self.img_rgb, axis=2).astype(np.uint8)
        self.h, self.w = self.gray.shape[:2]

    def test_true_pupil_candidate_receives_high_score(self):
        """A candidate aligned with the true pupil must score highly (> 0.60)."""
        gt_cx, gt_cy = self.gt["pupil_center"]
        gt_r = float(self.gt["pupil_radius"])

        true_cand = {
            "center": (gt_cx, gt_cy),
            "radius": gt_r,
            "rx": gt_r,
            "ry": gt_r,
            "angle": 0.0
        }

        score, breakdown = score_pupil_candidate(
            true_cand, self.gray, (self.h, self.w), return_breakdown=True
        )

        self.assertGreaterEqual(score, 0.60, f"True pupil candidate score {score} too low")
        self.assertGreater(breakdown["gradient"], 0.40)
        self.assertGreater(breakdown["contrast"], 0.40)
        self.assertGreater(breakdown["circularity"], 0.90)
        self.assertGreater(breakdown["plausibility"], 0.85)

    def test_eyelash_border_candidate_rejected_zero_score(self):
        """A candidate located near border or outside plausible bounds must receive score 0.0."""
        border_cand = {
            "center": (585, 220),  # near right border
            "radius": 85.0,        # giant radius
            "rx": 85.0,
            "ry": 80.0,
            "angle": 0.0
        }

        score, breakdown = score_pupil_candidate(
            border_cand, self.gray, (self.h, self.w), return_breakdown=True
        )

        self.assertEqual(score, 0.0, "Border candidate must be disqualified with score 0.0")
        self.assertIn("failures", breakdown)

    def test_calibrated_confidence_zeroed_by_hard_gate(self):
        """Confidence must be 0.0 if hard_constraint_gate is triggered."""
        conf, components = compute_calibrated_confidence(
            gradient_support=0.90,
            darkness_contrast=0.85,
            contour_completeness=0.88,
            inlier_ratio=0.80,
            plausibility_score=0.10,
            is_valid=False  # Gated
        )
        self.assertEqual(conf, 0.0)
        self.assertEqual(components["hard_constraint_gate"], 0.0)

    def test_calibrated_confidence_high_when_valid(self):
        """Confidence must be > 75% when evidence is strong and is_valid is True."""
        conf, components = compute_calibrated_confidence(
            gradient_support=0.85,
            darkness_contrast=0.80,
            contour_completeness=0.85,
            inlier_ratio=0.80,
            plausibility_score=1.0,
            is_valid=True
        )
        self.assertGreaterEqual(conf, 75.0)
        self.assertEqual(components["hard_constraint_gate"], 1.0)


if __name__ == "__main__":
    unittest.main()
