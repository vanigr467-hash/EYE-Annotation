"""
Backward-compatible wrapper for iris detection.
Exports IrisResult and detect_iris from src.iris_detector.
"""

from src.iris_detector import IrisResult, detect_iris

__all__ = ["IrisResult", "detect_iris"]
