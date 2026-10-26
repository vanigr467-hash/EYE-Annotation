import cv2

cap = cv2.VideoCapture("scratch/full_user_perfect.mp4")
for idx in [63, 75, 93, 150, 220]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"scratch/perfect_{idx}.png", frame)
cap.release()
print("Saved perfect sample frames")
