"""
Video Processing Pipeline with Multi-Stage Detection, Temporal Kalman Tracking,
and Key-Frame Analytics.
"""

import os
import time
from typing import Dict, Any, List, Tuple, Callable, Optional
import cv2
import numpy as np

from src.preprocessing import preprocess_eye_image
from src.pupil_detector import detect_pupil, PupilResult
from src.iris_detector import detect_iris, IrisResult
from src.eyelid_mask import detect_eyelid_mask
from src.validation import validate_detection_pair
from src.tracker import EyeTracker
from src.klt_video_tracker import KLTVideoTracker
from src.annotation import draw_automatic_annotations
from src.confidence import compute_overall_confidence
from utils.video_utils import frame_generator, read_video_metadata, write_video_file

MODE_FULL_DETECTION_INTERVAL = {
    "FAST": 10,
    "BALANCED": 4,
    "HIGH ACCURACY": 1
}


def process_eye_video(
    video_path: str,
    output_video_path: str,
    detection_mode: str = "BALANCED",
    frame_skip: int = 1,
    num_points: int = 64,
    theme: str = "Clinical Blueprint",
    smoothing_mode: str = "MEDIUM",
    sensitivity: float = 0.5,
    render_mode: str = "both",
    annotation_style: str = "Circles / Ellipses",
    show_values: bool = False,
    values_format: str = "Radius & Center (r, x, y)",
    custom_anchor: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Processes video frame-by-frame using EyeTracker temporal tracking, smoothing,
    and failure recovery. Supports dual output: standard overlay and pure mask on black canvas.
    """
    metadata = read_video_metadata(video_path)
    if not metadata["valid"]:
        raise ValueError(metadata["error"])

    # High-Performance Instant Real-Time Pipeline (60-120+ FPS)
    if detection_mode.upper().startswith("INSTANT") or detection_mode.upper().startswith("EASY"):
        from src.fast_video_annotator import process_eye_video_instant
        return process_eye_video_instant(
            video_path=video_path,
            output_video_path=output_video_path,
            num_points=num_points,
            theme=theme,
            frame_skip=frame_skip,
            render_mode=render_mode,
            annotation_style=annotation_style,
            show_values=show_values,
            values_format=values_format,
            custom_anchor=custom_anchor,
            progress_callback=progress_callback
        )

    fps = metadata.get("fps", 25.0)
    tracker = EyeTracker(fps=fps)
    klt_tracker = KLTVideoTracker(fps=fps)
    full_det_interval = MODE_FULL_DETECTION_INTERVAL.get(detection_mode.upper(), 4)

    annotated_frames = []
    annotation_only_frames = []
    original_frames = []
    frame_logs = []

    start_time = time.time()
    processed_count = 0
    success_count = 0
    predicted_count = 0
    failed_count = 0

    pupil_conf_list = []
    iris_conf_list = []
    overall_conf_list = []

    for frame_idx, timestamp, frame_rgb in frame_generator(video_path, frame_skip=frame_skip):
        processed_count += 1
        h, w = frame_rgb.shape[:2]

        is_tracking_active = (tracker.tracking_status == "TRACKING" and tracker.last_valid_state is not None)
        run_full_detection = (processed_count % full_det_interval == 0) or (not is_tracking_active)

        pupil_res = None
        iris_res = None
        offset_x, offset_y = 0, 0

        prep_dict = preprocess_eye_image(frame_rgb, clahe_clip=2.0, glint_removal=True)
        gray_curr = prep_dict.get("gray_inpainted", prep_dict.get("gray"))

        # Try KLT optical flow tracking on intermediate frames
        if not run_full_detection and is_tracking_active and klt_tracker.prev_gray is not None:
            klt_res = klt_tracker.track_frame_klt(gray_curr)
            if klt_res and klt_res.get("overall_confidence", 0.0) >= 45.0:
                p_c = klt_res["pupil_center"]
                p_r = klt_res["pupil_radius"]
                pupil_res = PupilResult(
                    center=p_c,
                    radius=p_r,
                    radius_x=klt_res.get("pupil_radius_x", p_r),
                    radius_y=klt_res.get("pupil_radius_y", p_r),
                    rotation=klt_res.get("pupil_rotation", 0.0),
                    confidence=klt_res["pupil_confidence"],
                    detected=True
                )
                i_c = klt_res["iris_center"]
                i_r = klt_res["iris_radius"]
                iris_res = IrisResult(
                    center=i_c,
                    radius=i_r,
                    radius_x=klt_res.get("iris_radius_x", i_r),
                    radius_y=klt_res.get("iris_radius_y", i_r),
                    rotation=klt_res.get("iris_rotation", 0.0),
                    confidence=klt_res["iris_confidence"],
                    detected=True
                )

        # Localized ROI search if KLT was not used or full detection is needed
        if (pupil_res is None or not pupil_res.detected) and not run_full_detection and is_tracking_active:
            roi_bbox = tracker.get_roi_bounding_box((h, w), margin_factor=3.2)
            if roi_bbox is not None:
                ymin, xmin, ymax, xmax = roi_bbox
                sub_frame = frame_rgb[ymin:ymax, xmin:xmax]
                offset_x, offset_y = xmin, ymin
                sub_prep = preprocess_eye_image(sub_frame, clahe_clip=2.0, glint_removal=True)

                p_sub = detect_pupil(sub_prep, sensitivity=sensitivity)
                if p_sub.detected:
                    p_full_center = (p_sub.center[0] + xmin, p_sub.center[1] + ymin)
                    p_full_ellipse = None
                    if p_sub.ellipse is not None:
                        ((ecx, ecy), (ew, eh), eang) = p_sub.ellipse
                        p_full_ellipse = ((ecx + xmin, ecy + ymin), (ew, eh), eang)

                    pupil_res = PupilResult(
                        center=p_full_center,
                        radius=p_sub.radius,
                        radius_x=p_sub.radius_x,
                        radius_y=p_sub.radius_y,
                        rotation=p_sub.rotation,
                        ellipse=p_full_ellipse,
                        circularity=p_sub.circularity,
                        darkness=p_sub.darkness,
                        confidence=p_sub.confidence,
                        detected=True
                    )

                    # Eyelid mask in sub-frame
                    eyelid_sub = detect_eyelid_mask(sub_prep["gray_inpainted"], p_sub.center, p_sub.radius)
                    i_sub = detect_iris(sub_prep, p_sub, eyelid_mask=eyelid_sub.valid_mask, sensitivity=sensitivity)

                    if i_sub.detected:
                        i_full_center = (i_sub.center[0] + xmin, i_sub.center[1] + ymin)
                        i_full_ellipse = None
                        if i_sub.ellipse is not None:
                            ((icx, icy), (iw, ih), iang) = i_sub.ellipse
                            i_full_ellipse = ((icx + xmin, icy + ymin), (iw, ih), iang)

                        iris_res = IrisResult(
                            center=i_full_center,
                            radius=i_sub.radius,
                            radius_x=i_sub.radius_x,
                            radius_y=i_sub.radius_y,
                            rotation=i_sub.rotation,
                            ellipse=i_full_ellipse,
                            confidence=i_sub.confidence,
                            quality_score=i_sub.quality_score,
                            inlier_ratio=i_sub.inlier_ratio,
                            detected=True
                        )
                else:
                    run_full_detection = True
            else:
                run_full_detection = True

        if run_full_detection or pupil_res is None or not pupil_res.detected:
            pupil_res = detect_pupil(prep_dict, sensitivity=sensitivity)

            if pupil_res.detected:
                eyelid_obj = detect_eyelid_mask(prep_dict["gray_inpainted"], pupil_res.center, pupil_res.radius)
                iris_res = detect_iris(prep_dict, pupil_res, eyelid_mask=eyelid_obj.valid_mask, sensitivity=sensitivity)
            else:
                iris_res = IrisResult(detected=False)

        # Update KLT tracker feature points whenever detector achieves good lock
        if pupil_res and iris_res and pupil_res.detected and iris_res.detected and pupil_res.confidence >= 60.0:
            klt_tracker.init_from_detection(gray_curr, pupil_res, iris_res)

        # Validate detection
        prev_state = tracker.last_valid_state
        val_res = validate_detection_pair(pupil_res, iris_res, (h, w), prev_state=prev_state)

        # Update Tracking State
        track_state = tracker.process_frame_tracking(pupil_res, iris_res, val_res)
        status = track_state["detection_status"]

        if status in ["DETECTED", "TRACKED"]:
            success_count += 1
        elif status == "PREDICTED":
            predicted_count += 1
        else:
            failed_count += 1

        p_conf = track_state["pupil_confidence"]
        i_conf = track_state["iris_confidence"]
        ov_conf = track_state["overall_confidence"]

        pupil_conf_list.append(p_conf)
        iris_conf_list.append(i_conf)
        overall_conf_list.append(ov_conf)

        # Build drawing results
        p_cen = track_state.get("pupil_center", (0, 0))
        p_rad = track_state.get("pupil_radius", 0.0)
        i_cen = track_state.get("iris_center", (0, 0))
        i_rad = track_state.get("iris_radius", 0.0)

        # Draw if valid coordinates exist, even if status is pending review
        has_draw_geom = (p_cen[0] > 0 and p_cen[1] > 0 and p_rad > 2)
        if has_draw_geom:
            draw_p_res = PupilResult(
                center=p_cen,
                radius=p_rad,
                confidence=max(30.0, p_conf),
                detected=True
            )
            irx = track_state.get("iris_radius_x", i_rad)
            iry = track_state.get("iris_radius_y", i_rad)
            irot = track_state.get("iris_rotation", 0.0)
            ell_tuple = (
                (float(i_cen[0]), float(i_cen[1])),
                (float(irx * 2), float(iry * 2)),
                float(irot)
            )
            draw_i_res = IrisResult(
                center=i_cen,
                radius=i_rad,
                radius_x=irx,
                radius_y=iry,
                rotation=irot,
                ellipse=ell_tuple,
                confidence=max(25.0, i_conf),
                detected=True
            )
        else:
            draw_p_res = PupilResult(center=(0, 0), radius=0.0, detected=False)
            draw_i_res = IrisResult(center=(0, 0), radius=0.0, detected=False)

        # Render visual annotations
        annotated_frame, points_data = draw_automatic_annotations(
            frame_rgb,
            draw_p_res,
            draw_i_res,
            num_points=num_points,
            theme=theme,
            show_pupil=True,
            show_iris=True,
            show_centers=True,
            show_points=True,
            show_values=show_values,
            boundary_style=annotation_style,
            values_format=values_format
        )

        # Overlay status badge
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx} | Status: {status} | Conf: {ov_conf:.0f}%",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

        annotated_frames.append(annotated_frame)
        original_frames.append(frame_rgb)

        if render_mode in ["both", "annotation_only"]:
            mask_frame, _ = draw_automatic_annotations(
                np.zeros_like(frame_rgb),
                draw_p_res,
                draw_i_res,
                num_points=num_points,
                theme=theme,
                show_pupil=True,
                show_iris=True,
                show_centers=True,
                show_points=True,
                show_values=show_values,
                boundary_style=annotation_style,
                values_format=values_format
            )
            cv2.putText(
                mask_frame,
                f"Frame: {frame_idx} | Status: {status} | Conf: {ov_conf:.0f}%",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            annotation_only_frames.append(mask_frame)

        frame_logs.append({
            "frame_number": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            "detection_status": status,
            "pupil_center_x": track_state["pupil_center"][0],
            "pupil_center_y": track_state["pupil_center"][1],
            "pupil_radius": round(track_state["pupil_radius"], 1),
            "iris_center_x": track_state["iris_center"][0],
            "iris_center_y": track_state["iris_center"][1],
            "iris_radius": round(track_state["iris_radius"], 1),
            "iris_radius_x": round(track_state.get("iris_radius_x", track_state["iris_radius"]), 1),
            "iris_radius_y": round(track_state.get("iris_radius_y", track_state["iris_radius"]), 1),
            "iris_rotation": round(track_state.get("iris_rotation", 0.0), 1),
            "pupil_confidence": round(p_conf, 1),
            "iris_confidence": round(i_conf, 1),
            "overall_confidence": round(ov_conf, 1)
        })

        if progress_callback and metadata["total_frames"] > 0:
            pct = float(processed_count / (metadata["total_frames"] / frame_skip))
            progress_callback(min(1.0, pct), f"Processing frame {processed_count} ({status})")

    elapsed_time = max(0.001, time.time() - start_time)
    processing_fps = round(processed_count / elapsed_time, 1)

    # Write output video file(s)
    out_fps = metadata["fps"] / float(frame_skip)
    base_p, ext_p = os.path.splitext(output_video_path)
    if not ext_p:
        ext_p = ".avi"
        output_video_path = base_p + ext_p

    written_video_path = output_video_path
    if render_mode != "annotation_only":
        write_video_file(annotated_frames, written_video_path, fps=out_fps)

    mask_video_path = f"{base_p}_annotation_only{ext_p}"
    if render_mode in ["both", "annotation_only"]:
        write_video_file(annotation_only_frames, mask_video_path, fps=out_fps)

    pupil_detection_rate = round(((success_count + predicted_count) / float(max(1, processed_count))) * 100.0, 1)
    iris_detection_rate = round((sum(1 for f in frame_logs if f["iris_radius"] > 0) / float(max(1, processed_count))) * 100.0, 1)

    avg_pupil_conf = round(float(np.mean(pupil_conf_list)) if pupil_conf_list else 0.0, 1)
    avg_iris_conf = round(float(np.mean(iris_conf_list)) if iris_conf_list else 0.0, 1)
    avg_overall_conf = round(float(np.mean(overall_conf_list)) if overall_conf_list else 0.0, 1)

    # Key-Frames Analysis
    sorted_by_conf = sorted(enumerate(frame_logs), key=lambda x: x[1]["overall_confidence"], reverse=True)
    best_10_indices = [idx for idx, _ in sorted_by_conf[:min(10, len(sorted_by_conf))]]
    worst_10_indices = [idx for idx, _ in sorted_by_conf[-min(10, len(sorted_by_conf)):]]

    primary_out = written_video_path if render_mode != "annotation_only" else mask_video_path

    return {
        "video_metadata": metadata,
        "output_video_path": primary_out,
        "annotated_video_path": written_video_path if render_mode != "annotation_only" else None,
        "annotation_only_video_path": mask_video_path if render_mode in ["both", "annotation_only"] else None,
        "render_mode": render_mode,
        "total_frames": metadata["total_frames"],
        "processed_frames": processed_count,
        "successful_detections": success_count,
        "predicted_frames": predicted_count,
        "failed_frames": failed_count,
        "pupil_detection_rate": pupil_detection_rate,
        "iris_detection_rate": iris_detection_rate,
        "avg_pupil_confidence": avg_pupil_conf,
        "avg_iris_confidence": avg_iris_conf,
        "avg_overall_confidence": avg_overall_conf,
        "processing_time_sec": round(elapsed_time, 2),
        "processing_fps": processing_fps,
        "original_frames": original_frames,
        "annotated_frames": annotated_frames,
        "annotation_only_frames": annotation_only_frames,
        "frame_logs": frame_logs,
        "best_10_indices": best_10_indices,
        "worst_10_indices": worst_10_indices
    }
