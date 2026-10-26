import cv2

cap = cv2.VideoCapture(r"C:\Users\vanig\AppData\Local\Temp\tmp0z2zsswh.mp4")
for i in range(60, 95, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, f = cap.read()
    if ret:
        cv2.imwrite(f"scratch/raw_frame_{i}.png", f)
cap.release()
print("Saved raw frames 60..90")
