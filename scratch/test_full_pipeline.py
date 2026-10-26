import cv2
import numpy as np

vid_path = r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4"
cap = cv2.VideoCapture(vid_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
border_mask = np.zeros((h, w), dtype=np.uint8)
border_mask[int(h*0.12):int(h*0.88), int(w*0.12):int(w*0.88)] = 255

def detect_pupil_eyelash_immune(frame_bgr, prev_cx=None, prev_cy=None, prev_r=42.0):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Eyelash-immune morphological open
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_disk)
    blurred = cv2.GaussianBlur(opened, (9, 9), 0)
    
    # 1. First try localized search if prior is available
    if prev_cx is not None and prev_cy is not None:
        win = int(prev_r * 3.5)
        x1 = max(0, int(prev_cx - win))
        y1 = max(0, int(prev_cy - win))
        x2 = min(w, int(prev_cx + win))
        y2 = min(h, int(prev_cy + win))
        
        sub = blurred[y1:y2, x1:x2]
        min_v, _, min_l, _ = cv2.minMaxLoc(sub)
        
        if min_v < 110:
            thresh = min(min_v + 28, 85)
            _, binary = cv2.threshold(sub, thresh, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_cnt = None
            best_circ = 0.0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 400 < area < 30000 and len(cnt) >= 5:
                    ((ecx, ecy), (ew, eh), eang) = cv2.fitEllipse(cnt)
                    circ = min(ew, eh) / max(ew, eh)
                    if circ > 0.55:
                        if circ > best_circ:
                            best_circ = circ
                            best_cnt = cnt
                            best_res = (ecx + x1, ecy + y1, ew/2.0, eh/2.0, eang)
                            
            if best_cnt is not None:
                return best_res, "TRACKED"

    # 2. Global search if local search failed or eye jumped
    masked_blurred = np.where(border_mask == 255, blurred, 255)
    min_v, _, min_l, _ = cv2.minMaxLoc(masked_blurred)
    
    if min_v < 115:
        thresh = min(min_v + 30, 85)
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_and(binary, border_mask)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 400 < area < 35000 and len(cnt) >= 5:
                ((ecx, ecy), (ew, eh), eang) = cv2.fitEllipse(cnt)
                circ = min(ew, eh) / max(ew, eh)
                if circ > 0.55:
                    # Score by circularity and closeness to center/prior
                    dist = np.hypot(ecx - (prev_cx or w/2), ecy - (prev_cy or h/2))
                    score = circ * 100.0 - dist * 0.1
                    candidates.append((score, (ecx, ecy, ew/2.0, eh/2.0, eang)))
                    
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1], "DETECTED"
            
    return None, "BLINK"

# Run across all frames
out_path = "scratch/full_user_perfect.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

curr_p = None
f_idx = 0
status_counts = {}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    prev_cx = curr_p[0] if curr_p else None
    prev_cy = curr_p[1] if curr_p else None
    prev_r = curr_p[2] if curr_p else 40.0
    
    res, status = detect_pupil_eyelash_immune(frame, prev_cx, prev_cy, prev_r)
    status_counts[status] = status_counts.get(status, 0) + 1
    
    if res is not None:
        if curr_p is None:
            curr_p = res
        else:
            # Smooth
            curr_p = (
                curr_p[0] * 0.3 + res[0] * 0.7,
                curr_p[1] * 0.3 + res[1] * 0.7,
                curr_p[2] * 0.4 + res[2] * 0.6,
                curr_p[3] * 0.4 + res[3] * 0.6,
                res[4]
            )
        disp_status = status
    else:
        disp_status = "BLINK (LOCKED)"
        
    pcx, pcy, prx, pry, pang = curr_p
    ir_r = prx * 2.6
    
    # Draw
    viz = frame.copy()
    cv2.circle(viz, (int(round(pcx)), int(round(pcy))), int(round(ir_r)), (0, 190, 255), 2, cv2.LINE_AA)
    cv2.ellipse(viz, (int(round(pcx)), int(round(pcy))), (int(round(prx)), int(round(pry))), pang, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(viz, (int(round(pcx)), int(round(pcy))), (0, 255, 0), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    
    color = (0, 255, 0) if disp_status in ["DETECTED", "TRACKED"] else (0, 165, 255)
    cv2.putText(viz, f"Frame {f_idx:3d} | {disp_status}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(viz, f"Frame {f_idx:3d} | {disp_status}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    
    writer.write(viz)
    f_idx += 1

cap.release()
writer.release()

print(f"Processed {f_idx} frames. Statuses:", status_counts)
