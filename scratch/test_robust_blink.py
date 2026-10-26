import cv2
import numpy as np

vid_path = r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4"
cap = cv2.VideoCapture(vid_path)

ret, f0 = cap.read()
h, w = f0.shape[:2]

# Test the robust blink-resistant pupil detector
def detect_pupil_easy(frame_bgr, prev_cx=None, prev_cy=None, prev_r=42.0):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Morphological open to completely erase thin eyelashes
    # A disk of radius 5 erases anything thinner than 10px (eyelashes) while keeping the 80px pupil intact!
    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_disk)
    
    # Also median blur to smooth glints
    blurred = cv2.medianBlur(cleaned, 9)
    
    # If we have a prior position, crop a search window (+- 120px)
    if prev_cx is not None and prev_cy is not None:
        win = int(prev_r * 3.0)
        x1 = max(0, int(prev_cx - win))
        y1 = max(0, int(prev_cy - win))
        x2 = min(w, int(prev_cx + win))
        y2 = min(h, int(prev_cy + win))
        search_region = blurred[y1:y2, x1:x2]
        ox, oy = x1, y1
    else:
        # Search center region (20% to 80% of image)
        x1, y1 = int(w * 0.15), int(h * 0.15)
        x2, y2 = int(w * 0.85), int(h * 0.85)
        search_region = blurred[y1:y2, x1:x2]
        ox, oy = x1, y1

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_region)
    
    # Darkest point in search region
    cand_cx = ox + min_loc[0]
    cand_cy = oy + min_loc[1]
    
    # Check if darkest point is in a blink (if image is mostly uniform or too bright)
    if min_val > 110:
        return None, "BLINK"
        
    # Local threshold around minimum
    thresh = min(min_val + 30, 85)
    _, binary = cv2.threshold(search_region, thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Find contour containing the min_loc
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_area = 0.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 45000:
            continue
        # Check if min_loc is inside or near this contour
        if cv2.pointPolygonTest(cnt, (float(min_loc[0]), float(min_loc[1])), True) >= -10:
            if area > best_area:
                best_area = area
                best_cnt = cnt
                
    if best_cnt is not None:
        M = cv2.moments(best_cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"] + ox
            cy = M["m01"] / M["m00"] + oy
            radius = np.sqrt(best_area / np.pi)
            
            # Check circularity
            perimeter = cv2.arcLength(best_cnt, True)
            if perimeter > 0:
                circ = 4 * np.pi * best_area / (perimeter ** 2)
                if circ < 0.35: # not circular (e.g. eyelid slit during blink)
                    return None, "BLINK"
                    
            return {"cx": cx, "cy": cy, "radius": radius}, "DETECTED"
            
    return None, "LOST"

# Run across all frames
f_idx = 0
cx, cy, r = 518.0, 532.0, 42.0

blink_count = 0
detected_count = 0

out_writer = cv2.VideoWriter("scratch/perfect_user_annotation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    pupil, status = detect_pupil_easy(frame, prev_cx=cx, prev_cy=cy, prev_r=r)
    
    if pupil is not None:
        detected_count += 1
        # Smoothly update
        cx = cx * 0.25 + pupil["cx"] * 0.75
        cy = cy * 0.25 + pupil["cy"] * 0.75
        r = r * 0.4 + pupil["radius"] * 0.6
        disp_status = "TRACKED"
    else:
        # Blink or lost - FREEZE position at last known good position!
        blink_count += 1
        disp_status = "BLINK (FROZEN)" if status == "BLINK" else "PREDICTED"
        
    ir_r = r * 2.7
    
    # Draw
    viz = frame.copy()
    cv2.circle(viz, (int(round(cx)), int(round(cy))), int(round(ir_r)), (0, 190, 255), 2, cv2.LINE_AA)
    cv2.circle(viz, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(viz, (int(round(cx)), int(round(cy))), (0, 255, 0), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    cv2.putText(viz, f"Frame {f_idx:3d} | {disp_status}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    out_writer.write(viz)
    
    if f_idx in [0, 63, 75, 80, 93, 150, 250]:
        cv2.imwrite(f"scratch/robust_frame_{f_idx}.png", viz)
        
    f_idx += 1

cap.release()
out_writer.release()

print(f"Processed {f_idx} frames: Detected={detected_count}, Blink/Frozen={blink_count}")
