import os
import sys
import cv2
import time
import numpy as np
from typing import Dict, Any, Optional, Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.video_processor import process_eye_video
from ml.inference.detector import detect_eye_boundaries

def process_video_with_tracking(
    video_path: str,
    output_video_path: str,
    detection_mode: str = 'BALANCED',
    frame_skip: int = 1,
    num_points: int = 64,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Process video with eye tracking and return per-frame tracking results along with the summary.
    Wraps the existing process_eye_video function but processes frames to build per-frame DB records.
    """
    
    result = {
        'summary': {},
        'frame_annotations': [],
        'annotated_frames': [],
        'original_frames': [],
        'output_video_path': output_video_path
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result['summary']['status'] = 'Failed to open video'
        return result
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    processed_count = 0
    successful_count = 0
    
    start_time = time.time()
    
    out_writer = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result['original_frames'].append(frame_rgb.copy())
            
            timestamp_sec = frame_idx / fps
            
            # Detect
            det_result = detect_eye_boundaries(
                frame_rgb,
                use_roi=True,
                use_eyelid_mask=True,
                num_points=num_points
            )
            
            annotated_rgb = det_result['annotated_image']
            result['annotated_frames'].append(annotated_rgb)
            
            # Write to output video
            if out_writer is None:
                h, w = annotated_rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_skip, (w, h))
                
            out_writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            
            # Record annotation
            frame_anno = {
                'frame_index': frame_idx,
                'timestamp_sec': timestamp_sec,
                'pupil': det_result['pupil'] if det_result['pupil']['detected'] else None,
                'iris': det_result['iris'] if det_result['iris']['detected'] else None,
                'overall_confidence': det_result['overall_confidence'],
                'detection_status': det_result['detection_status']
            }
            result['frame_annotations'].append(frame_anno)
            
            processed_count += 1
            if det_result['detection_status'] not in ['FAILED', 'ERROR']:
                successful_count += 1
                
            if progress_callback:
                progress_callback(frame_idx, total_frames)
                
            frame_idx += 1
            
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
            
    processing_time = time.time() - start_time
    
    result['summary'] = {
        'total_frames': total_frames,
        'processed_frames': processed_count,
        'successful_detections': successful_count,
        'processing_time': processing_time,
        'fps': processed_count / processing_time if processing_time > 0 else 0,
        'success_rate': successful_count / processed_count if processed_count > 0 else 0,
        'status': 'SUCCESS'
    }
    
    return result
