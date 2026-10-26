"""
Backward-compatible wrapper for eye tracking.
Exports EyeTracker from src.tracker.
"""

from src.tracker import EyeTracker, EyeKalmanFilter

__all__ = ["EyeTracker", "EyeKalmanFilter"]
