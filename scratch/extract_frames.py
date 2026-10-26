import cv2
import numpy as np

cap = cv2.VideoCapture("sample_videos/01_eyes.mp4")
frames = {}
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if idx in [0, 63, 93]:
        frames[idx] = frame.copy()
    idx += 1
cap.release()

print("Extracted frames:", list(frames.keys()))
for f_idx, f in frames.items():
    cv2.imwrite(f"scratch/frame_{f_idx}.png", f)
print("Saved frames 0, 63, 93 to scratch/")
