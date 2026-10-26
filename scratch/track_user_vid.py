import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
from src.fast_video_annotator import process_eye_video_instant

vid_path = r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4"
res = process_eye_video_instant(vid_path, "scratch/test_user_out.mp4")

logs = res["frame_logs"]
print(f"Total processed: {len(logs)}")
for f in logs[::10]:
    print(f"Frame {f['frame_number']:3d}: Pupil=({f['pupil_center_x']:5.1f}, {f['pupil_center_y']:5.1f}) r={f['pupil_radius']:4.1f} | Iris=({f['iris_center_x']:5.1f}, {f['iris_center_y']:5.1f}) r={f['iris_radius']:4.1f} | Status={f['detection_status']}")
