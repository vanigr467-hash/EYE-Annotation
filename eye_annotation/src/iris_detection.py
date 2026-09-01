import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.pupil_detection import PupilResult

class IrisResult:
    def __init__(
        self,
        center: Tuple[int, int],
        radius: float,
        ellipse: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None,
        edge_strength: float = 0.0,
        confidence: float = 0.0,
        mask: Optional[np.ndarray] = None,
        detected: bool = True,
        message: str = "Iris boundary detected successfully."
    ):
        self.center = center
        self.radius = radius
        self.ellipse = ellipse
        self.edge_strength = edge_strength
        self.confidence = confidence
        self.mask = mask
        self.detected = detected
        self.message = message

def detect_iris(
    prep_dict: Dict[str, Any],
    pupil_res: PupilResult,
    sensitivity: float = 0.5,
    fit_mode: str = "circle"
) -> IrisResult:
    """
    Detects iris boundary using pupil center anchor, Canny edge analysis, 
    radial gradient integration (Daugman approach), and Hough Circle fitting.
    """
    if not pupil_res.detected or pupil_res.radius <= 0:
        return IrisResult(
            center=(0, 0),
            radius=0.0,
            detected=False,
            message="Iris detection skipped: valid pupil anchor required."
        )

    gray = prep_dict["gray_inpainted"]
    clahe = prep_dict["clahe"]
    blurred = prep_dict["blurred"]
    h, w = gray.shape[:2]
    
    px, py = pupil_res.center
    pr = pupil_res.radius

    # Define iris radius search boundaries relative to pupil radius
    min_iris_r = int(pr * 1.6)
    max_iris_r = int(min(pr * 4.5, min(w, h) * 0.48))
    
    if min_iris_r >= max_iris_r or max_iris_r <= 5:
        return IrisResult(
            center=(px, py),
            radius=pr * 2.5,
            detected=False,
            message="Iris boundary uncertain: Pupil size relative to image resolution is outside optimal ratio."
        )

    # 1. Prepare Iris ROI Edge Map
    # Apply Canny edge detection on CLAHE image
    canny_thresh1 = int(30 * (1.0 - sensitivity))
    canny_thresh2 = int(100 * (1.0 - sensitivity))
    edges = cv2.Canny(clahe, canny_thresh1, canny_thresh2)

    # Mask out pupil interior to avoid inner pupil edge interference
    pupil_exclusion = np.zeros_like(edges)
    cv2.circle(pupil_exclusion, (px, py), int(pr * 1.15), 255, -1)
    edges[pupil_exclusion == 255] = 0

    # 2. Hough Circle Search centered around Pupil Anchor
    search_margin = max(10, int(pr * 0.4))
    hough_circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_iris_r,
        param1=50,
        param2=int(25 - 10 * sensitivity),
        minRadius=min_iris_r,
        maxRadius=max_iris_r
    )

    iris_candidates = []

    if hough_circles is not None:
        hough_circles = np.round(hough_circles[0, :]).astype(int)
        for (cx, cy, r) in hough_circles:
            # Check displacement from pupil center anchor
            dist_to_pupil = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist_to_pupil > search_margin:
                continue
                
            # Evaluate radial gradient strength along circle boundary (Daugman-like integral)
            edge_score = _evaluate_radial_gradient(clahe, cx, cy, r)
            
            # Score formula penalizes center displacement and favors radial edge strength
            disp_penalty = dist_to_pupil / float(search_margin)
            score = (edge_score * 0.8) + ((1.0 - disp_penalty) * 0.2)
            
            iris_candidates.append({
                "center": (cx, cy),
                "radius": float(r),
                "edge_strength": edge_score,
                "score": score
            })

    # 3. Radial Gradient Search fallback (if Hough produces no candidates)
    if not iris_candidates:
        r_range = np.linspace(min_iris_r, max_iris_r, int(max_iris_r - min_iris_r + 1))
        best_r = min_iris_r
        best_score = -1.0

        for r in r_range:
            score = _evaluate_radial_gradient(clahe, px, py, int(r))
            if score > best_score:
                best_score = score
                best_r = r

        iris_candidates.append({
            "center": (px, py),
            "radius": float(best_r),
            "edge_strength": max(0.1, best_score),
            "score": max(0.1, best_score)
        })

    # Select best iris candidate
    iris_candidates.sort(key=lambda x: x["score"], reverse=True)
    best_iris = iris_candidates[0]

    icx, icy = best_iris["center"]
    iradius = best_iris["radius"]
    edge_strength = best_iris["edge_strength"]

    # Generate iris mask
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_mask, (icx, icy), int(iradius), 255, -1)

    # Calculate confidence rating
    # Iris is confident if edge strength > threshold and ratio to pupil is reasonable (1.8 - 4.2)
    ratio = iradius / float(pr) if pr > 0 else 0
    ratio_score = 1.0 if (1.8 <= ratio <= 4.2) else 0.5
    confidence = float(np.clip((edge_strength * 0.6 + ratio_score * 0.4) * 100.0, 10.0, 95.0))

    detected = confidence >= 35.0
    msg = "Iris boundary detected successfully." if detected else "Warning: Iris boundary detection confidence is low. Please verify visual overlay."

    return IrisResult(
        center=(icx, icy),
        radius=iradius,
        edge_strength=edge_strength,
        confidence=confidence,
        mask=iris_mask,
        detected=detected,
        message=msg
    )

def _evaluate_radial_gradient(gray_img: np.ndarray, cx: int, cy: int, r: int, num_samples: int = 40) -> float:
    """Evaluates intensity contrast step across circle boundary at radius r (excluding top/bottom eyelids)."""
    h, w = gray_img.shape[:2]
    # Sample left and right arcs (ignoring upper and lower eyelids [-60 to +60 deg, 120 to 240 deg])
    angles = np.concatenate([
        np.linspace(-np.pi/3, np.pi/3, num_samples // 2),
        np.linspace(2*np.pi/3, 4*np.pi/3, num_samples // 2)
    ])
    
    r_inner = max(1, r - 3)
    r_outer = r + 3

    inner_vals = []
    outer_vals = []

    for a in angles:
        xi = int(cx + r_inner * np.cos(a))
        yi = int(cy + r_inner * np.sin(a))
        xo = int(cx + r_outer * np.cos(a))
        yo = int(cy + r_outer * np.sin(a))

        if 0 <= xi < w and 0 <= yi < h and 0 <= xo < w and 0 <= yo < h:
            inner_vals.append(gray_img[yi, xi])
            outer_vals.append(gray_img[yo, xo])

    if not inner_vals or not outer_vals:
        return 0.0

    # Iris intensity inside is darker than sclera outside (outer_val - inner_val > 0)
    contrast_diff = np.mean(outer_vals) - np.mean(inner_vals)
    return float(np.clip(contrast_diff / 50.0, 0.0, 1.0))
