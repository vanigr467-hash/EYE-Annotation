import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np

vid_path = r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4"
cap = cv2.VideoCapture(vid_path)

ret, f0 = cap.read()
h, w = f0.shape[:2]

# From frame 0 detection:
init_cx = 518.0
init_cy = 532.0
init_pr = 42.0
init_ir = init_pr * 2.6

curr_cx = init_cx
curr_cy = init_cy
curr_pr = init_pr

track_results = []
f_idx = 0

while True:
    if f_idx > 0:
        ret, frame = cap.read()
        if not ret: break
    else:
        frame = f0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # EASY METHOD: Local ROI search around previous center
    win_size = int(curr_pr * 2.5) # e.g. 100 px window
    x1 = max(0, int(curr_cx - win_size))
    y1 = max(0, int(curr_cy - win_size))
    x2 = min(w, int(curr_cx + win_size))
    y2 = min(h, int(curr_cy + win_size))

    roi_gray = gray[y1:y2, x1:x2]

    # Preprocess ROI: Median blur to eliminate thin eyelashes
    blurred = cv2.medianBlur(roi_gray, 7)

    # Pupil is the darkest circular region in the ROI
    min_val = np.min(blurred)
    # Threshold at min_val + 25 (only the dark pupil disc)
    thresh_val = min(min_val + 28, np.percentile(blurred, 15))
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Morphological open with circular disk to remove any eyelash tips
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours in ROI
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_dist = 999.0
    best_area = 0.0

    target_area = np.pi * (curr_pr ** 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < target_area * 0.3 or area > target_area * 2.5:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            mcx = M["m10"] / M["m00"] + x1
            mcy = M["m01"] / M["m00"] + y1
            dist = np.hypot(mcx - curr_cx, mcy - curr_cy)
            if dist < best_dist:
                best_dist = dist
                best_cnt = cnt
                best_center = (mcx, mcy)
                best_area = area

    if best_cnt is not None and best_dist < win_size * 0.8:
        # Smooth update
        curr_cx = curr_cx * 0.3 + best_center[0] * 0.7
        curr_cy = curr_cy * 0.3 + best_center[1] * 0.7
        est_r = np.sqrt(best_area / np.pi)
        curr_pr = curr_pr * 0.8 + est_r * 0.2
        status = "TRACKED"
    else:
        status = "PREDICTED"

    track_results.append((f_idx, curr_cx, curr_cy, curr_pr, curr_pr * 2.6, status))

    if f_idx in [0, 63, 93, 150, 200, 290]:
        # Save visualization
        viz = frame.copy()
        # Draw pupil
        cv2.circle(viz, (int(curr_cx), int(curr_cy)), int(curr_pr), (0, 255, 0), 2)
        # Draw iris
        cv2.circle(viz, (int(curr_cx), int(curr_cy)), int(curr_pr * 2.6), (0, 190, 255), 2)
        cv2.drawMarker(viz, (int(curr_cx), int(curr_cy)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(viz, f"Frame {f_idx} | {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(f"scratch/easy_result_{f_idx}.png", viz)

    f_idx += 1

cap.release()

print("Easy tracking completed for 299 frames!")
for r in track_results[::30]:
    print(f"Frame {r[0]:3d}: Pupil=({r[1]:5.1f}, {r[2]:5.1f}) r={r[3]:4.1f} | Status={r[5]}")
