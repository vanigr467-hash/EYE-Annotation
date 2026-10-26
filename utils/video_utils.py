import os
import cv2
import numpy as np
import imageio
from typing import Dict, Any, List, Tuple, Generator

def read_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extracts video metadata using OpenCV with fallback estimation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "valid": False,
            "error": f"Unable to open video file: {video_path}"
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or fps > 240:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If metadata props return 0/invalid, count frames manually
    if total_frames <= 0 or width <= 0 or height <= 0:
        count = 0
        w, h = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count == 0:
                h, w = frame.shape[:2]
            count += 1
        total_frames = count
        width, height = w, h
        cap.release()
        cap = cv2.VideoCapture(video_path)

    cap.release()

    duration_sec = total_frames / float(fps) if fps > 0 else 0.0

    return {
        "valid": True,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration_sec, 2),
        "aspect_ratio": round(width / float(height), 2) if height > 0 else 1.0
    }


def frame_generator(
    video_path: str,
    frame_skip: int = 1
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Generator yielding (frame_index, timestamp_sec, frame_rgb).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    idx = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        if idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = idx / float(fps)
            yield idx, timestamp, frame_rgb

        idx += 1

    cap.release()


def write_video_file(
    frames_rgb: List[np.ndarray],
    output_path: str,
    fps: float = 25.0
) -> str:
    """
    Writes a list of RGB frames into an MP4 (H.264) or AVI video file.
    Prefers imageio H.264 encoding for universal web browser playback.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not frames_rgb:
        return output_path

    h, w = frames_rgb[0].shape[:2]

    # For MP4, write via imageio H.264 for universal browser playback
    if output_path.endswith('.mp4'):
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=1)
            for frame_rgb in frames_rgb:
                writer.append_data(frame_rgb)
            writer.close()
            return output_path
        except Exception:
            pass

    # OpenCV fallback
    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        alt_path = output_path.rsplit('.', 1)[0] + '.avi'
        fourcc_alt = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(alt_path, fourcc_alt, fps, (w, h))
        output_path = alt_path

    for frame_rgb in frames_rgb:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path
