"""
Backward-compatible wrapper for pupil detection.
Exports PupilResult and detect_pupil from src.pupil_detector.
"""

from src.pupil_detector import PupilResult, detect_pupil

__all__ = ["PupilResult", "detect_pupil"]
