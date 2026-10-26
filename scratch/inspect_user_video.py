import cv2
import os

video_path = r"C:\Users\vanig\AppData\Local\Temp\annotated_eye_output_1792910766.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {video_path}: FPS={fps}, Total Frames={total}, W={w}, H={h}")

    for target in [0, 63, 93]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"scratch/user_annotated_{target}.png", frame)
            print(f"Saved user_annotated_{target}.png")
    cap.release()
else:
    print("Video does not exist")
