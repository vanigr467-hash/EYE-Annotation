import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class PupilResult:
    def __init__(
        self,
        center: Tuple[int, int],
        radius: float,
        ellipse: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None,
        circularity: float = 0.0,
        darkness: float = 0.0,
        confidence: float = 0.0,
        mask: Optional[np.ndarray] = None,
        detected: bool = True,
        message: str = "Pupil detected successfully."
    ):
        self.center = center
        self.radius = radius
        self.ellipse = ellipse # ((center_x, center_y), (axis_a, axis_b), angle)
        self.circularity = circularity
        self.darkness = darkness
        self.confidence = confidence
        self.mask = mask
        self.detected = detected
        self.message = message

def detect_pupil(
    prep_dict: Dict[str, Any],
    sensitivity: float = 0.5,
    fit_mode: str = "circle" # "circle" or "ellipse"
) -> PupilResult:
    """
    Detects pupil boundary using dark region analysis, adaptive thresholding, 
    contour circularity, and optional ellipse/circle fitting.
    """
    gray = prep_dict["gray_inpainted"]
    clahe = prep_dict["clahe"]
    blurred = prep_dict["blurred"]
    h, w = gray.shape[:2]
    img_area = h * w

    # Calculate expected radius bounds (e.g. 2% to 35% of image width)
    min_radius = max(8, int(w * 0.02))
    max_radius = int(w * 0.35)
    min_area = np.pi * (min_radius ** 2)
    max_area = np.pi * (max_radius ** 2)

    # 1. Multi-Threshold Candidate Extraction
    # Test dark intensity thresholds (e.g., 5th to 35th percentiles of brightness)
    p5 = np.percentile(blurred, 5)
    p25 = np.percentile(blurred, 25)
    
    threshold_vals = np.linspace(p5, p25 + 15 * sensitivity, 6)
    candidates = []

    for thresh in threshold_vals:
        _, binary = cv2.threshold(blurred, int(thresh), 255, cv2.THRESH_BINARY_INV)
        
        # Morphological opening to detach eyelashes/shadows & smooth contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
                
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < 0.45: # Filter out non-circular blobs
                continue
                
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx_i, cy_i = int(cx), int(cy)
            
            # Boundary sanity (center should not be at extreme edge)
            if cx_i < w * 0.1 or cx_i > w * 0.9 or cy_i < h * 0.1 or cy_i > h * 0.9:
                continue
                
            # Measure darkness inside candidate mask
            cnt_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_val = float(np.mean(gray[cnt_mask == 255]))
            
            # Calculate composite candidate score
            # Score favors: higher circularity, darker intensity, reasonable central position
            dist_center = np.sqrt((cx_i - w/2)**2 + (cy_i - h/2)**2) / (np.sqrt(w**2 + h**2) / 2)
            darkness_score = 1.0 - (mean_val / 255.0)
            
            score = (circularity * 0.45) + (darkness_score * 0.45) + ((1.0 - dist_center) * 0.1)
            
            candidates.append({
                "contour": cnt,
                "center": (cx_i, cy_i),
                "radius": r,
                "area": area,
                "circularity": circularity,
                "darkness": mean_val,
                "darkness_score": darkness_score,
                "score": score,
                "mask": cnt_mask
            })

    # 2. Hough Circle Verification (Fallback / Boost)
    hough_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w * 0.2,
        param1=50,
        param2=int(30 - 10 * sensitivity),
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if hough_circles is not None:
        hough_circles = np.round(hough_circles[0, :]).astype(int)
        for (hx, hy, hr) in hough_circles:
            # Measure darkness in Hough circle
            y_grid, x_grid = np.ogrid[:h, :w]
            h_mask = (x_grid - hx)**2 + (y_grid - hy)**2 <= hr**2
            if np.any(h_mask):
                mean_val = float(np.mean(gray[h_mask]))
                darkness_score = 1.0 - (mean_val / 255.0)
                dist_center = np.sqrt((hx - w/2)**2 + (hy - h/2)**2) / (np.sqrt(w**2 + h**2) / 2)
                score = (0.7 * darkness_score) + ((1.0 - dist_center) * 0.3)
                
                # Check if matches any contour candidate
                matching = False
                for c in candidates:
                    dist = np.sqrt((c["center"][0] - hx)**2 + (c["center"][1] - hy)**2)
                    if dist < hr * 0.5:
                        c["score"] += 0.2 # Boost candidate score
                        matching = True
                        break
                if not matching and darkness_score > 0.65:
                    candidates.append({
                        "contour": None,
                        "center": (hx, hy),
                        "radius": float(hr),
                        "area": np.pi * hr**2,
                        "circularity": 0.85,
                        "darkness": mean_val,
                        "darkness_score": darkness_score,
                        "score": score,
                        "mask": h_mask.astype(np.uint8) * 255
                    })

    if not candidates:
        return PupilResult(
            center=(w // 2, h // 2),
            radius=0.0,
            detected=False,
            message="Unable to confidently detect pupil. Please try a clearer eye image."
        )

    # Sort candidates by composite score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    # Fit circle or ellipse
    best_cnt = best["contour"]
    fitted_ellipse = None

    if fit_mode == "ellipse" and best_cnt is not None and len(best_cnt) >= 5:
        try:
            ellipse_param = cv2.fitEllipse(best_cnt)
            ((ecx, ecy), (e_a, e_b), angle) = ellipse_param
            fitted_ellipse = ((ecx, ecy), (e_a, e_b), angle)
            center = (int(ecx), int(ecy))
            radius = (e_a + e_b) / 4.0 # Average radius
        except Exception:
            center = best["center"]
            radius = best["radius"]
    else:
        center = best["center"]
        radius = best["radius"]

    # Generate pupil binary mask
    pupil_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(pupil_mask, center, int(radius), 255, -1)

    confidence = float(np.clip(best["score"] * 100.0, 10.0, 99.0))

    return PupilResult(
        center=center,
        radius=radius,
        ellipse=fitted_ellipse,
        circularity=best["circularity"],
        darkness=best["darkness"],
        confidence=confidence,
        mask=pupil_mask,
        detected=True,
        message="Pupil detected successfully."
    )
