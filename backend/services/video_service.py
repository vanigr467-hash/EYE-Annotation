import os
from typing import List, Dict

def extract_frames(video_path: str, fps: float, storage_service) -> List[Dict]:
    """Extract frames from a video at a specified frame rate."""
    import cv2
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0
        
    frame_interval = int(round(orig_fps / fps)) if fps > 0 else 1
    if frame_interval < 1:
        frame_interval = 1
        
    extracted_frames = []
    frame_idx = 0
    saved_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / orig_fps
            
            temp_path = f"temp_frame_{saved_idx}.jpg"
            cv2.imwrite(temp_path, frame)
            
            with open(temp_path, "rb") as f:
                # Assuming save_file signature for standard use
                storage_url = storage_service.save_file(f"frame_{saved_idx}.jpg", f.read(), folder="frames")
            
            os.remove(temp_path)
            
            extracted_frames.append({
                "frame_index": saved_idx,
                "timestamp_sec": timestamp_sec,
                "storage_url": storage_url,
                "width": frame.shape[1],
                "height": frame.shape[0]
            })
            saved_idx += 1
            
        frame_idx += 1
        
    cap.release()
    return extracted_frames
