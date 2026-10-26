import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from src.fast_video_annotator import _fast_detect_anchor

vid_path = r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4"
cap = cv2.VideoCapture(vid_path)
ret, f0 = cap.read()
cap.release()

print("Frame 0 shape:", f0.shape)
cv2.imwrite("scratch/user_raw_frame0.png", f0)

pupil, iris = _fast_detect_anchor(f0)
print("Pupil detected on frame 0:", pupil)
print("Iris detected on frame 0:", iris)
