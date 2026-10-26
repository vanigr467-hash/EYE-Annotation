import cv2
import numpy as np

# Load raw frame 93
f93 = cv2.imread("scratch/raw_frame_90.png")
if f93 is None:
    cap = cv2.VideoCapture(r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 93)
    ret, f93 = cap.read()
    cap.release()

gray = cv2.cvtColor(f93, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

# Method to detect pupil anywhere in the frame:
# 1. Morphological open to remove eyelashes
kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_disk)

# 2. Smooth
blurred = cv2.GaussianBlur(opened, (9, 9), 0)

# Mask out frame borders (10% on each side)
mask = np.zeros_like(gray)
mask[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)] = 255
masked_blurred = np.where(mask == 255, blurred, 255)

# Find darkest point
min_val, _, min_loc, _ = cv2.minMaxLoc(masked_blurred)
print(f"Frame 93 global darkest point: {min_loc} with val {min_val}")

# Threshold around minimum
thresh = min_val + 25
_, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
binary = cv2.bitwise_and(binary, mask)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 500 < area < 20000:
        ((ecx, ecy), (ew, eh), eang) = cv2.fitEllipse(cnt)
        circ = min(ew, eh) / max(ew, eh)
        if circ > 0.6:
            print(f"Found pupil candidate: center=({ecx:.1f}, {ecy:.1f}), r={ew/2:.1f}, circ={circ:.2f}")
            viz = f93.copy()
            cv2.ellipse(viz, ((ecx, ecy), (ew, eh), eang), (0, 255, 0), 2)
            cv2.circle(viz, (int(ecx), int(ecy)), int(ew * 1.3), (0, 190, 255), 2)
            cv2.imwrite("scratch/frame_93_global_pupil.png", viz)
