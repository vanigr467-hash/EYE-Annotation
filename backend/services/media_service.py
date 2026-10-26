import os
from typing import Tuple, Dict

def get_image_dimensions(file_path: str) -> Tuple[int, int]:
    """Get width and height of an image."""
    import cv2
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image: {file_path}")
    return img.shape[1], img.shape[0]

def get_video_metadata(file_path: str) -> Dict:
    """Get metadata for a video file."""
    import cv2
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {file_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration
    }
