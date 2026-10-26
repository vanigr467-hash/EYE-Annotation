"""
Named Constants Configuration for Eye Annotation Platform.
Defines geometric plausibility limits, candidate generation sweeps,
and confidence calibration weights.
"""

# =========================================================
# Geometric Plausibility Constraints
# =========================================================

# Pupil Constraints
PUPIL_BORDER_MARGIN_RATIO = 0.06      # Center must be >= 0.06 * min(H, W) from any border
PUPIL_MIN_RADIUS_RATIO = 0.015        # Radius >= 0.015 * min(H, W)
PUPIL_MAX_RADIUS_RATIO = 0.18         # Radius <= 0.18 * min(H, W)
PUPIL_MIN_AXIS_RATIO = 0.55           # min(rx, ry) / max(rx, ry) >= 0.55
PUPIL_MIN_CONTAINMENT_RATIO = 0.92    # >= 92% of perimeter must lie inside frame

# Iris Constraints
IRIS_MIN_RADIUS_RATIO = 0.10          # Radius >= 0.10 * min(H, W)
IRIS_MAX_RADIUS_RATIO = 0.45          # Radius <= 0.45 * min(H, W)
IRIS_MIN_AXIS_RATIO = 0.65            # min(rx, ry) / max(rx, ry) >= 0.65
IRIS_MIN_CONTAINMENT_RATIO = 0.85     # >= 85% of perimeter must lie inside frame

# Pupil-Iris Relationship Constraints
MAX_PUPIL_IRIS_CENTER_OFFSET_RATIO = 0.35  # distance(p_center, i_center) <= 0.35 * r_iris
MIN_PUPIL_IRIS_RADIUS_RATIO = 0.15         # 0.15 <= r_pupil / r_iris
MAX_PUPIL_IRIS_RADIUS_RATIO = 0.75         # r_pupil / r_iris <= 0.75

# =========================================================
# Pupil Candidate Generation & Scoring
# =========================================================
CANDIDATE_PERCENTILES = [5, 8, 12, 18, 25, 35]
MIN_CONTOUR_POINTS_FOR_FIT = 5

WEIGHT_RADIAL_GRADIENT = 0.35
WEIGHT_DARKNESS_CONTRAST = 0.20
WEIGHT_CIRCULARITY = 0.15
WEIGHT_CONTOUR_COMPLETENESS = 0.15
WEIGHT_PLAUSIBILITY_PRIOR = 0.15

# =========================================================
# Confidence Recalibration Weights
# =========================================================
CONF_WEIGHT_GRADIENT = 0.30
CONF_WEIGHT_CONTRAST = 0.20
CONF_WEIGHT_COMPLETENESS = 0.15
CONF_WEIGHT_INLIER_RATIO = 0.15
CONF_WEIGHT_PLAUSIBILITY = 0.20

# =========================================================
# Temporal Tracking & Gating
# =========================================================
MAX_TRACKING_JUMP_RATIO = 0.50        # Max displacement allowed vs r_iris before gating
MAX_CONSECUTIVE_GATED_FRAMES = 5      # Consecutive gated frames before track declared LOST
BLINK_INTENSITY_DROP_THRESH = 0.40    # Palpebral aperture drop indicator
