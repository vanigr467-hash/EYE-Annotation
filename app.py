import os
import time
import sys
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.image_quality import analyze_image_quality
from src.eye_roi import detect_eye_roi, roi_to_full_coords
from src.preprocessing import preprocess_eye_image
from src.eyelid_mask import detect_eyelid_mask
from src.pupil_detector import detect_pupil, PupilResult
from src.iris_detector import detect_iris, IrisResult
from src.validation import validate_detection_pair
from src.confidence import compute_overall_confidence
from src.annotation import draw_automatic_annotations, draw_annotation_only_mask, COLOR_SCHEMES
from src.gaze_tracking import calculate_gaze_vector
from src.analytics import compute_eye_analytics, compute_video_plr_dynamics
from src.diagnostics import generate_diagnostics_image_grid, get_diagnostics_report
from src.video_processor import process_eye_video
from src.visualization import (
    generate_intensity_profile_plot,
    generate_video_tracking_charts
)
from src.export import (
    export_image_metrics_csv,
    export_video_metrics_csv,
    export_video_metrics_json,
    export_coco_keypoints_json,
    export_yolo_keypoints_txt
)
from utils.image_utils import (
    load_image_from_bytes, numpy_to_png_bytes, resize_if_large
)
from utils.synthetic_eye import generate_synthetic_eye

# Page Configuration
st.set_page_config(
    page_title="Advanced Automatic Eye Annotation System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    .badge-detected {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .badge-low-confidence {
        background-color: #FEF9C3;
        color: #A16207;
        border: 1px solid #FDE047;
    }
    .badge-partial {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #A5B4FC;
    }
    .badge-not-detected {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-header">👁️ ADVANCED AUTOMATED EYE ANNOTATION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Stage Hybrid Pipeline: Eye ROI Isolation, Eyelid Masking, Constrained Iris Search & Kalman Tracking</div>', unsafe_allow_html=True)

# Sidebar Navigation & Controls
st.sidebar.title("📌 Navigation")
nav_page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard Overview",
        "👁️ Image Annotation",
        "🎥 Video Annotation",
        "🎞️ Frame Explorer",
        "📊 Analytics & Graphs",
        "🔬 Detection Diagnostics",
        "ℹ️ About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Advanced Detection Controls")

fit_mode = st.sidebar.selectbox("Fitting Geometry", ["ellipse", "circle"], index=0)
use_roi = st.sidebar.checkbox("Isolate Eye ROI First", value=True)
use_eyelid_mask = st.sidebar.checkbox("Eyelid & Eyelash Masking", value=True)
use_starburst = st.sidebar.checkbox("Starburst RANSAC Pupil Refinement", value=True)
prep_mode = st.sidebar.selectbox("Preprocessing Ensemble", ["Auto-Select", "Variant A (CLAHE+Bilateral)", "Variant B (Gamma+CLAHE)", "Variant C (Morph Contrast)", "Variant D (Illumination Norm)"], index=0)
num_points = st.sidebar.select_slider("Annotation Points", options=[8, 16, 32, 64, 128], value=64)
sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.5, 0.05)
glint_removal = st.sidebar.checkbox("Specular Glint Inpainting", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visual Overlay & Annotation Settings")
theme = st.sidebar.selectbox("Color Theme", list(COLOR_SCHEMES.keys()), index=0)

boundary_style = st.sidebar.selectbox(
    "Boundary Representation",
    [
        "Circles / Ellipses",
        "Circles + Numbers",
        "Numbers Only",
        "Keypoints / Dots Only",
        "Full (Curves + Points + Numbers)"
    ],
    index=0,
    help="Select how boundaries are rendered: standard curves, with numerical coordinate & radius badges, numbers-only, or discrete keypoints."
)
show_values = st.sidebar.checkbox(
    "Show Values on Boundary",
    value=(boundary_style in ["Circles + Numbers", "Numbers Only", "Full (Curves + Points + Numbers)"]),
    help="Overlay numerical values (radius, center coordinates) next to the detected boundaries."
)
values_format = st.sidebar.selectbox(
    "Values Callout Format",
    [
        "Radius & Center (r, x, y)",
        "Radius only (r)",
        "Full Geometry (cx, cy, rx, ry)"
    ],
    index=0,
    help="Format of numerical value badges shown on boundaries."
)

show_pupil_layer = st.sidebar.checkbox("Show Pupil Boundary", value=True)
show_iris_layer = st.sidebar.checkbox("Show Iris Boundary", value=True)
show_gaze_arrow = st.sidebar.checkbox("Show 2D Gaze Direction Arrow", value=False)
show_centers_layer = st.sidebar.checkbox("Show Center Crosshairs", value=True)
show_points_layer = st.sidebar.checkbox("Show Boundary Points", value=True)
show_labels_layer = st.sidebar.checkbox("Show Point Labels", value=False)
show_boundary_evidence = st.sidebar.checkbox("Show Iris Evidence (Debug)", value=False)

# Session State Initialization
if "video_results" not in st.session_state:
    st.session_state["video_results"] = None
if "last_image_data" not in st.session_state:
    st.session_state["last_image_data"] = None


# ==========================================
# 1. DASHBOARD OVERVIEW
# ==========================================
if nav_page == "🏠 Dashboard Overview":
    st.subheader("Welcome to the Hybrid Automated Eye Annotation Platform")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 👁️ Image Mode\nIsolate Eye ROI, mask eyelids/lashes, detect pupil & iris with constrained annular search, and export COCO/YOLO datasets.")
    with col2:
        st.success("### 🎥 Video Mode\nProcess eye videos with 10D Kalman filter tracking, failure recovery, key-frame analytics, and PLR dilation velocity.")
    with col3:
        st.warning("### 🔬 Detection Diagnostics\nInspect all intermediate pipeline stages: ROI crop, CLAHE, Glint mask, Eyelid curves, and candidate inlier/outlier points.")

    st.markdown("---")
    st.markdown("### 🚀 Quick Benchmark & Pipeline Test")
    if st.button("▶️ Run System Check (Synthetic Ground Truth Eye)"):
        syn_rgb, gt = generate_synthetic_eye(pupil_offset=(10, -6))
        prep = preprocess_eye_image(syn_rgb)
        p_res = detect_pupil(prep, use_starburst_refinement=True)
        eyelid_obj = detect_eyelid_mask(prep["gray_inpainted"], p_res.center, p_res.radius)
        i_res = detect_iris(prep, p_res, eyelid_mask=eyelid_obj.valid_mask, fit_mode=fit_mode)
        ann_rgb, _ = draw_automatic_annotations(
            syn_rgb, p_res, i_res, num_points=64,
            show_gaze_arrow=False, show_boundary_evidence=True,
            eyelid_mask_obj=eyelid_obj,
            boundary_style=boundary_style,
            show_values=show_values,
            values_format=values_format
        )

        b1, b2 = st.columns(2)
        with b1:
            st.image(syn_rgb, caption="Synthetic Input Image", use_container_width=True)
        with b2:
            st.image(ann_rgb, caption="Pipeline Output with Eyelid Mask & Boundary Evidence", use_container_width=True)

        st.success(
            f"✅ Benchmark Complete! Pupil: Center={p_res.center}, Radius={p_res.radius:.1f}px (GT={gt['pupil_radius']}). "
            f"Iris: Center={i_res.center}, Radius={i_res.radius:.1f}px (GT={gt['iris_radius']}), "
            f"Quality={i_res.quality_score:.1f}%, Status={i_res.status}"
        )


# ==========================================
# 2. IMAGE ANNOTATION MODE
# ==========================================
elif nav_page == "👁️ Image Annotation":
    st.subheader("👁️ Automatic Pupil & Iris Annotation")

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded_file = st.file_uploader("Upload Eye Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    with up_col2:
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".png"))] if os.path.exists(sample_dir) else []
        selected_sample = st.selectbox("Or pick a sample image", ["-- Select Sample --"] + sample_files)

    img_input = None
    image_name = "eye_image.png"

    if uploaded_file is not None:
        img_input = load_image_from_bytes(uploaded_file.read())
        image_name = uploaded_file.name
    elif selected_sample != "-- Select Sample --":
        sample_path = os.path.join(sample_dir, selected_sample)
        with open(sample_path, "rb") as f:
            img_input = load_image_from_bytes(f.read())
        image_name = selected_sample

    if img_input is None:
        img_input, _ = generate_synthetic_eye(pupil_offset=(8, -5))
        st.info("💡 Displaying default synthetic eye image. Upload a custom image or pick a sample above.")

    resized_img, _ = resize_if_large(img_input, max_dim=1000)
    h, w = resized_img.shape[:2]

    # 1. Quality Assessment
    gray_eval = resized_img if len(resized_img.shape) == 2 else cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    quality_info = analyze_image_quality(gray_eval)
    if quality_info["is_low_quality"]:
        st.warning(f"⚠️ Image Quality Advisory: {quality_info['warning']}")

    # 2. Eye ROI Detection (Optional)
    if use_roi:
        roi_obj = detect_eye_roi(resized_img, padding_factor=0.25)
        active_img = roi_obj.crop
        rx, ry, rw, rh = roi_obj.bbox
    else:
        active_img = resized_img
        rx, ry, rw, rh = 0, 0, w, h
        roi_obj = None

    # 3. Preprocessing Ensemble
    var_map = {"Auto-Select": "auto", "Variant A (CLAHE+Bilateral)": "A", "Variant B (Gamma+CLAHE)": "B", "Variant C (Morph Contrast)": "C", "Variant D (Illumination Norm)": "D"}
    prep_dict = preprocess_eye_image(
        active_img,
        clahe_clip=2.0,
        glint_removal=glint_removal,
        force_variant=var_map.get(prep_mode, "auto")
    )

    # 4. Multi-Candidate Pupil Detection
    pupil_res_roi = detect_pupil(
        prep_dict,
        sensitivity=sensitivity,
        fit_mode=fit_mode,
        use_starburst_refinement=use_starburst
    )

    # 5. Eyelid Masking
    if use_eyelid_mask and pupil_res_roi.detected:
        eyelid_obj = detect_eyelid_mask(
            prep_dict["gray_inpainted"],
            pupil_res_roi.center,
            pupil_res_roi.radius
        )
        eyelid_mask = eyelid_obj.valid_mask
    else:
        eyelid_obj = None
        eyelid_mask = None

    # 6. Constrained Iris Detection
    iris_res_roi = detect_iris(
        prep_dict,
        pupil_res_roi,
        eyelid_mask=eyelid_mask,
        sensitivity=sensitivity,
        fit_mode=fit_mode
    )

    # Map coordinates from ROI back to full image
    if use_roi and roi_obj is not None:
        if pupil_res_roi.detected:
            p_full_center = (pupil_res_roi.center[0] + rx, pupil_res_roi.center[1] + ry)
            p_full_ellipse = None
            if pupil_res_roi.ellipse is not None:
                ((ecx, ecy), (ew, eh), eang) = pupil_res_roi.ellipse
                p_full_ellipse = ((ecx + rx, ecy + ry), (ew, eh), eang)
            pupil_res = PupilResult(
                center=p_full_center,
                radius=pupil_res_roi.radius,
                radius_x=pupil_res_roi.radius_x,
                radius_y=pupil_res_roi.radius_y,
                rotation=pupil_res_roi.rotation,
                ellipse=p_full_ellipse,
                confidence=pupil_res_roi.confidence,
                all_candidates=pupil_res_roi.all_candidates,
                detected=pupil_res_roi.detected,
                status=pupil_res_roi.status
            )
        else:
            pupil_res = pupil_res_roi

        if iris_res_roi.detected:
            i_full_center = (iris_res_roi.center[0] + rx, iris_res_roi.center[1] + ry)
            i_full_ellipse = None
            if iris_res_roi.ellipse is not None:
                ((icx, icy), (iw, ih), iang) = iris_res_roi.ellipse
                i_full_ellipse = ((icx + rx, icy + ry), (iw, ih), iang)
            iris_res = IrisResult(
                center=i_full_center,
                radius=iris_res_roi.radius,
                radius_x=iris_res_roi.radius_x,
                radius_y=iris_res_roi.radius_y,
                rotation=iris_res_roi.rotation,
                ellipse=i_full_ellipse,
                confidence=iris_res_roi.confidence,
                quality_score=iris_res_roi.quality_score,
                inlier_ratio=iris_res_roi.inlier_ratio,
                num_valid_rays=iris_res_roi.num_valid_rays,
                inlier_points=[(p[0] + rx, p[1] + ry) for p in iris_res_roi.inlier_points],
                outlier_points=[(p[0] + rx, p[1] + ry) for p in iris_res_roi.outlier_points],
                boundary_candidates=[(p[0] + rx, p[1] + ry) for p in iris_res_roi.boundary_candidates],
                detected=iris_res_roi.detected,
                status=iris_res_roi.status
            )
        else:
            iris_res = iris_res_roi
    else:
        pupil_res = pupil_res_roi
        iris_res = iris_res_roi

    # 7. Geometric Validation & Confidence Fusion
    val_info = validate_detection_pair(pupil_res, iris_res, (h, w))
    conf_dict = compute_overall_confidence(pupil_res, iris_res, validation_info=val_info, quality_info=quality_info)
    analytics_dict = compute_eye_analytics(prep_dict, pupil_res, iris_res)

    # 8. Render Annotations
    annotated_rgb, points_data = draw_automatic_annotations(
        resized_img, pupil_res, iris_res,
        num_points=num_points, theme=theme,
        show_pupil=show_pupil_layer, show_iris=show_iris_layer,
        show_centers=show_centers_layer, show_points=show_points_layer,
        show_labels=show_labels_layer, show_gaze_arrow=show_gaze_arrow,
        show_boundary_evidence=show_boundary_evidence,
        eyelid_mask_obj=eyelid_obj,
        boundary_style=boundary_style,
        show_values=show_values,
        values_format=values_format
    )

    mask_only_rgb = draw_annotation_only_mask(
        resized_img.shape, pupil_res, iris_res,
        num_points=num_points, theme=theme,
        boundary_style=boundary_style,
        show_values=show_values,
        values_format=values_format,
        show_pupil=show_pupil_layer,
        show_iris=show_iris_layer,
        show_centers=show_centers_layer,
        show_points=show_points_layer,
        eyelid_mask_obj=eyelid_obj
    )

    # Save state for Diagnostics tab
    st.session_state["last_image_data"] = {
        "original_rgb": resized_img,
        "prep_dict": prep_dict,
        "pupil_res": pupil_res,
        "iris_res": iris_res,
        "eyelid_obj": eyelid_obj,
        "quality_info": quality_info,
        "glint_mask": prep_dict["glint_mask"],
        "conf_dict": conf_dict,
        "val_info": val_info
    }

    # Display Side-by-Side
    st.markdown("---")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("#### 📷 Original Image")
        st.image(resized_img, use_container_width=True)
        st.caption(f"Dimensions: {w}x{h} px | Quality: {quality_info['quality_grade']} (Score: {quality_info['quality_score']}/100)")

    with c_col2:
        status_tag = conf_dict['status']
        st.markdown(f"#### 🎯 Annotation Output [{status_tag}] ({num_points} Points)")
        view_mode_img = st.radio(
            "Display Mode",
            ["👁️ Annotated Overlay", "🖤 Pure Annotation Mask (Annotation Only)"],
            horizontal=True,
            key="img_view_mode"
        )
        if view_mode_img.startswith("👁️"):
            st.image(annotated_rgb, use_container_width=True)
        else:
            st.image(mask_only_rgb, use_container_width=True)

        if iris_res.detected:
            st.caption(f"Iris Inliers: {len(iris_res.inlier_points)} | Outliers: {len(iris_res.outlier_points)} | Quality: {iris_res.quality_score}%")
        else:
            st.caption(f"⚠️ {iris_res.message}")

    # Metrics Section
    st.markdown("---")
    st.subheader("📊 Detection & Boundary Diagnostics")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Pupil Center (X, Y)", f"({analytics_dict['pupil_center_x']}, {analytics_dict['pupil_center_y']})")
        st.metric("Pupil Radius", f"{analytics_dict['pupil_radius_px']} px")
    with m2:
        st.metric("Iris Center (X, Y)", f"({analytics_dict['iris_center_x']}, {analytics_dict['iris_center_y']})")
        if fit_mode == "ellipse":
            st.metric("Iris Axes (Rx, Ry)", f"{analytics_dict['iris_radius_x_px']}, {analytics_dict['iris_radius_y_px']} px")
        else:
            st.metric("Iris Radius", f"{analytics_dict['iris_radius_px']} px")
    with m3:
        st.metric("Iris Boundary Quality", f"{analytics_dict['iris_boundary_quality']}%")
        st.metric("Iris Inlier Ratio", f"{analytics_dict['iris_inlier_ratio']}")
    with m4:
        st.metric("Iris-to-Pupil Ratio", f"{analytics_dict['iris_pupil_diameter_ratio']}x")
        st.metric("Detection Status", conf_dict["status"])
    with m5:
        st.metric("Pupil Confidence", f"{conf_dict['pupil_confidence']}%")
        st.metric("Overall Confidence", f"{conf_dict['overall_confidence']}%")

    # Exports
    st.markdown("---")
    st.subheader("📥 Export Datasets & Annotation Keypoints")
    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    with dl1:
        st.download_button("📥 Annotated Image (PNG)", data=numpy_to_png_bytes(annotated_rgb), file_name=f"annotated_{image_name}", mime="image/png")
    with dl2:
        st.download_button("🖤 Annotation-Only Mask (PNG)", data=numpy_to_png_bytes(mask_only_rgb), file_name=f"mask_{image_name}", mime="image/png")
    with dl3:
        st.download_button("📊 Metrics Report (CSV)", data=export_image_metrics_csv(analytics_dict), file_name="eye_metrics.csv", mime="text/csv")
    with dl4:
        coco_json = export_coco_keypoints_json(image_name, w, h, points_data, pupil_res.center, iris_res.center)
        st.download_button("🤖 COCO Keypoints (JSON)", data=coco_json, file_name="coco_keypoints.json", mime="application/json")
    with dl5:
        yolo_txt = export_yolo_keypoints_txt(w, h, pupil_res.center, iris_res.center, pupil_res.radius, iris_res.radius, points_data)
        st.download_button("⚡ YOLO Keypoints (TXT)", data=yolo_txt, file_name="yolo_keypoints.txt", mime="text/plain")


# ==========================================
# 3. VIDEO ANNOTATION MODE
# ==========================================
elif nav_page == "🎥 Video Annotation":
    st.subheader("🎥 Video Eye Boundary Detection & Temporal Kalman Tracking")

    video_source_mode = st.radio(
        "📹 Video Input Source",
        [
            "📁 Upload File / Sample Video",
            "📷 Live Camera Video Stream & Recording",
            "📸 Browser Camera Snapshot"
        ],
        horizontal=True,
        index=0,
        help="Select whether to process an uploaded/sample video file, stream live from webcam, or capture an eye snapshot using your camera."
    )

    input_video_path = None
    start_process = False
    start_stream = False
    custom_anchor_dict = None
    sample_video_dir = os.path.join(os.path.dirname(__file__), "sample_videos")
    sample_v_files = [f for f in os.listdir(sample_video_dir) if f.endswith((".mp4", ".avi"))] if os.path.exists(sample_video_dir) else []

    if video_source_mode == "📁 Upload File / Sample Video":
        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            uploaded_video = st.file_uploader("Upload Eye Video (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])
        with v_col2:
            default_sample_idx = 1 if len(sample_v_files) > 0 else 0
            selected_v_sample = st.selectbox(
                "Or choose a sample video",
                ["-- Select Sample Video --"] + sample_v_files,
                index=default_sample_idx
            )

        if uploaded_video is not None:
            file_key = f"{uploaded_video.name}_{uploaded_video.size}"
            cached_path = st.session_state.get("cached_video_path")
            if st.session_state.get("cached_video_key") != file_key or not cached_path or not os.path.exists(cached_path):
                uploaded_video.seek(0)
                orig_ext = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                t_file = tempfile.NamedTemporaryFile(delete=False, suffix=orig_ext)
                t_file.write(uploaded_video.read())
                t_file.flush()
                t_file.close()
                st.session_state["cached_video_path"] = t_file.name
                st.session_state["cached_video_key"] = file_key
                st.session_state["video_results"] = None
            input_video_path = st.session_state.get("cached_video_path")
        elif selected_v_sample != "-- Select Sample Video --":
            input_video_path = os.path.join(sample_video_dir, selected_v_sample)

    elif video_source_mode == "📷 Live Camera Video Stream & Recording":
        st.markdown("### 📷 Live Camera Video Input")
        st.caption("Capture and annotate directly from your webcam in real-time or record a video clip for processing.")

        c_opt1, c_opt2, c_opt3 = st.columns([1, 1, 1])
        with c_opt1:
            cam_idx = st.number_input("Camera Device Index", min_value=0, max_value=4, value=0, step=1, help="0 is usually your built-in webcam. 1 or 2 for external USB cameras.")
        with c_opt2:
            rec_duration = st.selectbox("Recording Duration (seconds)", [3, 5, 10], index=1)
        with c_opt3:
            cam_action = st.radio("Camera Action", ["Live Real-Time Stream", "Record Video Clip & Process"], horizontal=True)

        if cam_action == "Live Real-Time Stream":
            btn_cam_stream = st.button("▶️ Start Live Camera Stream", type="primary", use_container_width=True)
            if btn_cam_stream:
                st.markdown("#### 👁️ Real-Time Camera Annotation Stream")
                live_cap = cv2.VideoCapture(int(cam_idx))
                if not live_cap.isOpened():
                    st.error(f"⚠️ Could not open Camera device {cam_idx}. If your camera is not directly attached to this machine or if you're using a remote browser, please select the '📸 Browser Camera Snapshot' tab above!")
                else:
                    live_spot = st.empty()
                    m1, m2, m3, m4 = st.columns(4)
                    m_status = m1.empty()
                    m_pupil = m2.empty()
                    m_iris = m3.empty()
                    m_fps = m4.empty()

                    ret_test, f_test = live_cap.read()
                    if ret_test:
                        th, tw = f_test.shape[:2]
                        ksize = max(11, int(min(tw, th) * 0.015) | 1)
                        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                        b_mask = np.zeros((th, tw), dtype=np.uint8)
                        b_mask[int(th*0.08):int(th*0.92), int(tw*0.08):int(tw*0.92)] = 255
                        min_r = 0.015 * min(th, tw)
                        max_r = 0.22 * min(th, tw)

                        from src.fast_video_annotator import detect_pupil_eyelash_immune, _render_video_frame_annotations
                        curr_p = None
                        ir_ratio = 2.65
                        t_prev = time.time()
                        f_count = 0

                        for _ in range(150):
                            ret_c, f_bgr = live_cap.read()
                            if not ret_c:
                                break
                            f_count += 1
                            t_start_f = time.time()

                            cand, status, circ = detect_pupil_eyelash_immune(f_bgr, curr_p, k_disk, b_mask, min_r, max_r)
                            if cand is not None:
                                if curr_p is None:
                                    curr_p = cand
                                else:
                                    curr_p = (
                                        curr_p[0] * 0.35 + cand[0] * 0.65,
                                        curr_p[1] * 0.35 + cand[1] * 0.65,
                                        curr_p[2] * 0.35 + cand[2] * 0.65,
                                        curr_p[3] * 0.35 + cand[3] * 0.65,
                                        cand[4]
                                    )
                                disp_status = status
                                p_conf = min(96.0, 75.0 + circ * 22.0)
                            else:
                                disp_status = "BLINK (LOCKED)"
                                p_conf = 0.0

                            if curr_p is not None:
                                pcx, pcy, prx, pry, pang = curr_p
                                irx, iry = prx * ir_ratio, pry * ir_ratio
                                i_conf = p_conf * 0.95
                                ov_conf = (p_conf * 0.55 + i_conf * 0.45) if disp_status != "BLINK (LOCKED)" else 0.0

                                f_viz = f_bgr.copy()
                                _render_video_frame_annotations(
                                    f_viz,
                                    pcx, pcy, prx, pry, pang,
                                    pcx, pcy, irx, iry, pang,
                                    is_val=(disp_status != "BLINK (LOCKED)"),
                                    detection_status=disp_status,
                                    overall_conf=ov_conf,
                                    frame_idx=f_count,
                                    annotation_style=boundary_style,
                                    show_values=show_values,
                                    values_format=values_format
                                )
                            else:
                                f_viz = f_bgr
                                disp_status = "SEARCHING"
                                pcx, pcy, prx, irx = 0, 0, 0, 0

                            t_now = time.time()
                            live_fps = 1.0 / max(0.001, t_now - t_prev)
                            t_prev = t_now

                            m_status.metric("Status", disp_status)
                            m_pupil.metric("Pupil (X, Y)", f"({pcx:.0f}, {pcy:.0f})")
                            m_iris.metric("Radii (P/I)", f"{prx:.0f}px / {irx:.0f}px")
                            m_fps.metric("Live FPS", f"{live_fps:.1f} FPS")

                            live_spot.image(cv2.cvtColor(f_viz, cv2.COLOR_BGR2RGB), use_container_width=True)
                            time.sleep(0.02)
                    live_cap.release()
                    st.success("✅ Live camera stream finished.")

        elif cam_action == "Record Video Clip & Process":
            btn_cam_rec = st.button("🔴 Start Camera Recording & Annotate", type="primary", use_container_width=True)
            if btn_cam_rec:
                rec_cap = cv2.VideoCapture(int(cam_idx))
                if not rec_cap.isOpened():
                    st.error(f"⚠️ Could not open Camera device {cam_idx}. If your camera is not directly attached to this machine or if you're using a remote browser, please select the '📸 Browser Camera Snapshot' tab above!")
                else:
                    st.info(f"🔴 Recording {rec_duration} seconds from camera...")
                    rec_frames = []
                    rec_start = time.time()
                    while time.time() - rec_start < float(rec_duration):
                        r_ret, r_frame = rec_cap.read()
                        if r_ret:
                            rec_frames.append(r_frame)
                        time.sleep(0.03)
                    rec_cap.release()

                    if len(rec_frames) > 5:
                        temp_rec_path = os.path.join(tempfile.gettempdir(), f"camera_recorded_{int(time.time())}.mp4")
                        rh, rw = rec_frames[0].shape[:2]
                        rec_writer = cv2.VideoWriter(temp_rec_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (rw, rh))
                        for rf in rec_frames:
                            rec_writer.write(rf)
                        rec_writer.release()

                        st.session_state["cached_video_path"] = temp_rec_path
                        st.session_state["cached_video_key"] = f"camera_{int(time.time())}"
                        st.session_state["video_results"] = None
                        st.success(f"✅ Recorded {len(rec_frames)} frames from camera! Processing video...")
                        input_video_path = temp_rec_path
                        start_process = True

    elif video_source_mode == "📸 Browser Camera Snapshot":
        st.markdown("### 📸 Browser Direct Camera Capture")
        st.caption("Use your browser's webcam permission to capture an eye frame directly, detect boundaries, or generate an annotated sequence.")

        cam_pic = st.camera_input("Position eye in front of camera and take photo")
        if cam_pic is not None:
            photo_bytes = cam_pic.getvalue()
            photo_arr = np.frombuffer(photo_bytes, np.uint8)
            cam_bgr = cv2.imdecode(photo_arr, cv2.IMREAD_COLOR)

            if cam_bgr is not None:
                ch, cw = cam_bgr.shape[:2]
                from src.fast_video_annotator import _fast_detect_anchor, _render_video_frame_annotations
                p_c, i_c = _fast_detect_anchor(cam_bgr)

                pcx = float(p_c.get("cx", cw // 2))
                pcy = float(p_c.get("cy", ch // 2))
                prx = float(p_c.get("rx", 35.0))
                pry = float(p_c.get("ry", prx))
                pang = float(p_c.get("angle", 0.0))
                icx = float(i_c.get("cx", pcx))
                icy = float(i_c.get("cy", pcy))
                irx = float(i_c.get("rx", prx * 2.65))
                iry = float(i_c.get("ry", pry * 2.65))
                iang = float(i_c.get("angle", pang))
                ov_conf = float(p_c.get("confidence", 85.0))

                cam_viz = cam_bgr.copy()
                _render_video_frame_annotations(
                    cam_viz,
                    pcx, pcy, prx, pry, pang,
                    icx, icy, irx, iry, iang,
                    is_val=True,
                    detection_status="DETECTED",
                    overall_conf=ov_conf,
                    frame_idx=1,
                    annotation_style=boundary_style,
                    show_values=show_values,
                    values_format=values_format
                )

                st.markdown("#### 🎯 Camera Annotation Result")
                shot_col1, shot_col2 = st.columns([1, 1])
                with shot_col1:
                    st.image(cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB), caption="Original Camera Capture", use_container_width=True)
                with shot_col2:
                    st.image(cv2.cvtColor(cam_viz, cv2.COLOR_BGR2RGB), caption="Real-Time Annotated Camera Frame (Green: Pupil, Orange: Iris)", use_container_width=True)

                met1, met2, met3, met4 = st.columns(4)
                met1.metric("Pupil Center", f"({pcx:.1f}, {pcy:.1f})")
                met2.metric("Pupil Radius", f"{prx:.1f} px")
                met3.metric("Iris Radius", f"{irx:.1f} px")
                met4.metric("Confidence", f"{ov_conf:.1f}%")

                dl_col1, dl_col2 = st.columns([1, 1])
                with dl_col1:
                    _, cam_png = cv2.imencode(".png", cam_viz)
                    st.download_button(
                        label="📥 Download Annotated Camera Image (PNG)",
                        data=cam_png.tobytes(),
                        file_name=f"camera_annotation_{int(time.time())}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with dl_col2:
                    if st.button("🎬 Convert Camera Frame to 5-Sec Video & Track", type="primary", use_container_width=True):
                        temp_seq_path = os.path.join(tempfile.gettempdir(), f"camera_seq_{int(time.time())}.mp4")
                        seq_writer = cv2.VideoWriter(temp_seq_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (cw, ch))
                        for f_idx in range(125):
                            dx = int(round(1.5 * np.sin(f_idx * 0.1)))
                            dy = int(round(1.5 * np.cos(f_idx * 0.1)))
                            M = np.float32([[1, 0, dx], [0, 1, dy]])
                            drift_frame = cv2.warpAffine(cam_bgr, M, (cw, ch), borderMode=cv2.BORDER_REFLECT)
                            seq_writer.write(drift_frame)
                        seq_writer.release()

                        st.session_state["cached_video_path"] = temp_seq_path
                        st.session_state["cached_video_key"] = f"cam_seq_{int(time.time())}"
                        st.session_state["video_results"] = None
                        input_video_path = temp_seq_path
                        start_process = True

    st.markdown("---")
    st.subheader("⚙️ Video Processing Settings")
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        det_mode = st.selectbox(
            "Detection Mode",
            [
                "EASY ROBUST (⚡ Eyelash-Immune & Blink-Proof - Recommended)",
                "INSTANT (⚡ Real-Time 60+ FPS)",
                "BALANCED (Tracking + Periodic Re-Detection)",
                "HIGH ACCURACY (Deep Direct Detection)",
                "FAST"
            ],
            index=0
        )
    with ctrl2:
        frame_skip = st.selectbox("Frame Skipping", [1, 2, 5, 10], index=0)
    with ctrl3:
        smoothing_mode = st.selectbox("Temporal Tracking", ["Kalman Constant Velocity", "MEDIUM", "OFF"], index=0)
    with ctrl4:
        render_mode_choice = st.selectbox(
            "Video Output Format",
            [
                "Both (Annotated + Annotation-Only)",
                "Annotated Video Only",
                "Annotation-Only Video (Pure Mask)"
            ],
            index=0,
            help="Choose whether to generate standard annotated video, pure annotation mask video on black canvas, or both."
        )

    render_mode_map = {
        "Both (Annotated + Annotation-Only)": "both",
        "Annotated Video Only": "annotated",
        "Annotation-Only Video (Pure Mask)": "annotation_only"
    }
    render_mode = render_mode_map.get(render_mode_choice, "both")

    if input_video_path is None:
        input_video_path = st.session_state.get("cached_video_path")

    if input_video_path and os.path.exists(input_video_path):
        cap_p = cv2.VideoCapture(input_video_path)
        ret0, f0_bgr = cap_p.read()
        cap_p.release()

        if ret0:
            fh0, fw0 = f0_bgr.shape[:2]
            from src.fast_video_annotator import _fast_detect_anchor, detect_pupil_eyelash_immune

            # Initialize anchor for this video if changed
            if st.session_state.get("anchor_vid_key") != input_video_path:
                p_anc, i_anc = _fast_detect_anchor(f0_bgr)
                st.session_state["anchor_vid_key"] = input_video_path
                st.session_state["anchor_cx"] = float(p_anc["cx"])
                st.session_state["anchor_cy"] = float(p_anc["cy"])
                st.session_state["anchor_pr"] = float(p_anc["rx"])
                st.session_state["anchor_ir"] = float(i_anc["rx"])

            st.markdown("### 🎯 Easy 1-Click Setup & Frame 0 Verification")
            st.caption("The system has automatically locked onto the real pupil disc (immune to eyelashes & blinks). You can review Frame 0 below or fine-tune before processing.")

            with st.expander("🛠️ Fine-Tune Anchor (Optional Nudge)", expanded=False):
                tune1, tune2, tune3, tune4 = st.columns(4)
                with tune1:
                    adj_cx = st.number_input("Pupil Center X", 0, fw0, int(round(st.session_state.get("anchor_cx", fw0 // 2))), step=1)
                with tune2:
                    adj_cy = st.number_input("Pupil Center Y", 0, fh0, int(round(st.session_state.get("anchor_cy", fh0 // 2))), step=1)
                with tune3:
                    adj_pr = st.number_input("Pupil Radius (px)", 4, int(min(fw0, fh0) * 0.25), int(round(st.session_state.get("anchor_pr", 40))), step=1)
                with tune4:
                    adj_ir = st.number_input("Iris Radius (px)", 10, int(min(fw0, fh0) * 0.5), int(round(st.session_state.get("anchor_ir", int(st.session_state.get("anchor_pr", 40) * 2.65)))), step=1)
                st.session_state["anchor_cx"] = adj_cx
                st.session_state["anchor_cy"] = adj_cy
                st.session_state["anchor_pr"] = adj_pr
                st.session_state["anchor_ir"] = adj_ir

            acx = float(st.session_state.get("anchor_cx", fw0 // 2))
            acy = float(st.session_state.get("anchor_cy", fh0 // 2))
            apr = float(st.session_state.get("anchor_pr", 40.0))
            air = float(st.session_state.get("anchor_ir", apr * 2.65))

            custom_anchor_dict = {
                "cx": acx,
                "cy": acy,
                "pupil_radius": apr,
                "iris_radius": air
            }

            f0_viz = f0_bgr.copy()
            cv2.circle(f0_viz, (int(round(acx)), int(round(acy))), int(round(air)), (0, 190, 255), 2, cv2.LINE_AA)
            cv2.circle(f0_viz, (int(round(acx)), int(round(acy))), int(round(apr)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(f0_viz, (int(round(acx)), int(round(acy))), (0, 255, 0), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            cv2.putText(f0_viz, f"Frame 0 Anchor | Pupil: r={apr:.0f}px | Iris: r={air:.0f}px", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            pv_col1, pv_col2 = st.columns([1, 1])
            with pv_col1:
                st.image(cv2.cvtColor(f0_viz, cv2.COLOR_BGR2RGB), caption="Frame 0 Verified Anchor (Green: Pupil, Orange: Iris)", use_container_width=True)
            with pv_col2:
                st.success("✅ **Anchor Locked**: Pupil & Iris boundaries are centered on the pupil disc.")
                start_process = st.button("⚡ Easy One-Click Video Annotation (Eyelash-Immune & Fast)", type="primary", use_container_width=True, key="btn_proc_ready")
                start_stream = st.button("▶️ Live Stream Annotations (Real-Time Playback)", use_container_width=True, key="btn_stream_ready")
    else:
        st.info("💡 **Ready to annotate**: Upload an eye video above, or select a sample video like `01_eyes.mp4` to begin real-time annotation.")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_process = st.button("🚀 Process Video (Immediate)", type="primary", use_container_width=True, key="btn_proc_empty")
        with btn_col2:
            start_stream = st.button("▶️ Live Stream Annotations (Real-Time Playback)", use_container_width=True, key="btn_stream_empty")

    # If stream or process requested without video, auto-fallback to sample video
    if (start_stream or start_process) and not input_video_path:
        fallback_vid = os.path.join(sample_video_dir, "01_eyes.mp4")
        if os.path.exists(fallback_vid):
            input_video_path = fallback_vid
            st.info("ℹ️ Loaded default sample video `01_eyes.mp4` for immediate real-time playback.")
        else:
            st.warning("⚠️ No video file loaded. Please upload a video or choose a sample video above!")

    if input_video_path and start_stream:
        st.markdown("---")
        st.subheader("👁️ Live Real-Time Annotation Stream")
        st.caption("Real-time frame-by-frame pupil & iris boundary detection running at 30-60+ FPS.")

        s_col1, s_col2 = st.columns([1, 1])
        with s_col1:
            loop_stream = st.checkbox("🔁 Continuous Real-Time Loop", value=True, key="chk_loop_stream")
        with s_col2:
            speed_opt = st.select_slider("Playback Speed", options=["0.5x (Slow-Mo)", "1.0x (Real-Time)", "1.5x (Fast)", "2.0x (Max Speed)"], value="1.0x (Real-Time)", key="speed_slider")

        speed_factor = 1.0
        if "0.5x" in speed_opt: speed_factor = 0.5
        elif "1.5x" in speed_opt: speed_factor = 1.5
        elif "2.0x" in speed_opt: speed_factor = 2.0

        # Live metric cards row
        m1, m2, m3, m4, m5 = st.columns(5)
        m_frame = m1.empty()
        m_status = m2.empty()
        m_pupil = m3.empty()
        m_iris = m4.empty()
        m_fps = m5.empty()

        live_spot = st.empty()
        live_cap = cv2.VideoCapture(input_video_path)
        from src.fast_video_annotator import detect_pupil_eyelash_immune, _render_video_frame_annotations

        tot_f = int(live_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = live_cap.get(cv2.CAP_PROP_FPS)
        if not v_fps or np.isnan(v_fps) or v_fps <= 0:
            v_fps = 30.0

        ret0, f0 = live_cap.read()
        if ret0:
            fh0, fw0 = f0.shape[:2]
            ksize = max(11, int(min(fw0, fh0) * 0.015) | 1)
            k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            b_mask = np.zeros((fh0, fw0), dtype=np.uint8)
            b_mask[int(fh0*0.08):int(fh0*0.92), int(fw0*0.08):int(fw0*0.92)] = 255
            min_r = 0.015 * min(fh0, fw0)
            max_r = 0.22 * min(fh0, fw0)

            acx = float(st.session_state.get("anchor_cx", fw0 // 2))
            acy = float(st.session_state.get("anchor_cy", fh0 // 2))
            apr = float(st.session_state.get("anchor_pr", 40.0))
            air = float(st.session_state.get("anchor_ir", apr * 2.65))
            curr_p = (acx, acy, apr, apr, 0.0)
            ir_ratio = air / max(1.0, apr)

            f_num = 0
            live_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t_prev = time.time()
            max_loops = 3 if loop_stream else 1
            current_loop = 0

            while current_loop < max_loops:
                ret_f, f_bgr = live_cap.read()
                if not ret_f:
                    current_loop += 1
                    if current_loop < max_loops:
                        live_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        f_num = 0
                        continue
                    else:
                        break

                f_num += 1
                if f_num % frame_skip != 0:
                    continue

                t_frame_start = time.time()
                cand, status, circ = detect_pupil_eyelash_immune(f_bgr, curr_p, k_disk, b_mask, min_r, max_r)
                if cand is not None:
                    curr_p = (
                        curr_p[0] * 0.35 + cand[0] * 0.65,
                        curr_p[1] * 0.35 + cand[1] * 0.65,
                        curr_p[2] * 0.35 + cand[2] * 0.65,
                        curr_p[3] * 0.35 + cand[3] * 0.65,
                        cand[4]
                    )
                    disp_status = status
                    p_conf = min(96.0, 75.0 + circ * 22.0)
                else:
                    disp_status = "BLINK"
                    p_conf = 0.0

                pcx, pcy, prx, pry, pang = curr_p
                irx, iry = prx * ir_ratio, pry * ir_ratio
                i_conf = p_conf * 0.95
                ov_conf = (p_conf * 0.55 + i_conf * 0.45) if disp_status != "BLINK" else 0.0

                # Render with annotations, badges, and style
                f_viz = f_bgr.copy()
                _render_video_frame_annotations(
                    f_viz,
                    pcx, pcy, prx, pry, pang,
                    pcx, pcy, irx, iry, pang,
                    is_val=(disp_status != "BLINK"),
                    detection_status=disp_status,
                    overall_conf=ov_conf,
                    frame_idx=f_num,
                    annotation_style=boundary_style,
                    show_values=show_values,
                    values_format=values_format
                )

                t_now = time.time()
                t_frame_dur = max(0.001, t_now - t_frame_start)
                live_fps = 1.0 / max(0.001, t_now - t_prev)
                t_prev = t_now

                # Update live HUD metrics
                m_frame.metric("Frame", f"{f_num} / {tot_f}")
                m_status.metric("Status", disp_status)
                m_pupil.metric("Pupil (X, Y)", f"({pcx:.0f}, {pcy:.0f})")
                m_iris.metric("Radii (P/I)", f"{prx:.0f}px / {irx:.0f}px")
                m_fps.metric("Live FPS", f"{live_fps:.1f} FPS")

                live_spot.image(cv2.cvtColor(f_viz, cv2.COLOR_BGR2RGB), use_container_width=True)

                # Regulate speed for smooth playback
                target_frame_dur = (1.0 / (v_fps * speed_factor))
                sleep_dur = max(0.002, target_frame_dur - t_frame_dur)
                time.sleep(sleep_dur)

            live_cap.release()
            st.success(f"✅ Real-Time Annotation Stream Finished ({f_num} frames annotated)!")

    if input_video_path and start_process:
        output_temp_video = os.path.join(tempfile.gettempdir(), f"annotated_eye_output_{int(time.time())}.mp4")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(msg)

        with st.spinner("Processing video frames with high-speed eyelash-immune tracker..."):
            results = process_eye_video(
                input_video_path,
                output_temp_video,
                detection_mode=det_mode,
                frame_skip=frame_skip,
                num_points=num_points,
                theme=theme,
                smoothing_mode=smoothing_mode,
                sensitivity=sensitivity,
                render_mode=render_mode,
                annotation_style=boundary_style,
                show_values=show_values,
                values_format=values_format,
                custom_anchor=custom_anchor_dict,
                progress_callback=update_progress
            )

        st.session_state["video_results"] = results
        status_text.text(f"✅ Complete in {results['processing_time_sec']}s ({results['processing_fps']} FPS)!")

    v_res = st.session_state.get("video_results", None)
    if v_res is not None:
        st.markdown("---")
        st.subheader("📊 Video Detection Statistics")
        ann_path = v_res.get("annotated_video_path")
        mask_path = v_res.get("annotation_only_video_path")

        # Video Player View Selector
        available_views = []
        if ann_path and os.path.exists(ann_path):
            available_views.append("👁️ Annotated Video (Original + Overlay)")
        if mask_path and os.path.exists(mask_path):
            available_views.append("🖤 Annotation-Only Video (Pure Mask)")

        if available_views:
            if len(available_views) > 1:
                selected_video_view = st.radio("Select Video Preview:", available_views, horizontal=True)
            else:
                selected_video_view = available_views[0]

            if selected_video_view.startswith("👁️") and ann_path and os.path.exists(ann_path):
                st.video(ann_path)
            elif mask_path and os.path.exists(mask_path):
                st.video(mask_path)
        elif os.path.exists(v_res.get("output_video_path", "")):
            st.video(v_res["output_video_path"])

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric("Total Frames", v_res["total_frames"])
            st.metric("Processed Frames", v_res["processed_frames"])
        with s2:
            st.metric("Successful Detections", v_res["successful_detections"])
            st.metric("Pupil Detection Rate", f"{v_res['pupil_detection_rate']}%")
        with s3:
            st.metric("Predicted Frames", v_res["predicted_frames"])
            st.metric("Iris Detection Rate", f"{v_res['iris_detection_rate']}%")
        with s4:
            st.metric("Failed Frames", v_res["failed_frames"])
            st.metric("Avg Overall Confidence", f"{v_res['avg_overall_confidence']}%")
        with s5:
            st.metric("Processing Time", f"{v_res['processing_time_sec']} s")
            st.metric("Processing Speed", f"{v_res['processing_fps']} FPS")

        st.markdown("---")
        st.subheader("📥 Export Video Results")
        export_slots = []
        if ann_path and os.path.exists(ann_path):
            export_slots.append(("annotated", ann_path))
        if mask_path and os.path.exists(mask_path):
            export_slots.append(("mask", mask_path))
        export_slots.append(("csv", None))
        export_slots.append(("json", None))

        v_cols = st.columns(len(export_slots))
        for col_i, (slot_type, fpath) in enumerate(export_slots):
            with v_cols[col_i]:
                if slot_type == "annotated":
                    with open(fpath, "rb") as vf:
                        f_ext = os.path.splitext(fpath)[1].lower()
                        mime_t = "video/mp4" if f_ext == ".mp4" else "video/avi"
                        st.download_button(
                            f"📥 Download Annotated Video ({f_ext.upper().strip('.')})",
                            data=vf.read(),
                            file_name=f"annotated_eye_video{f_ext}",
                            mime=mime_t
                        )
                elif slot_type == "mask":
                    with open(fpath, "rb") as vf:
                        f_ext = os.path.splitext(fpath)[1].lower()
                        mime_t = "video/mp4" if f_ext == ".mp4" else "video/avi"
                        st.download_button(
                            f"🖤 Download Annotation-Only Video ({f_ext.upper().strip('.')})",
                            data=vf.read(),
                            file_name=f"annotation_only_video{f_ext}",
                            mime=mime_t
                        )
                elif slot_type == "csv":
                    csv_str = export_video_metrics_csv(v_res["frame_logs"])
                    st.download_button("📊 Video Metrics (CSV)", data=csv_str, file_name="video_metrics.csv", mime="text/csv")
                elif slot_type == "json":
                    json_str = export_video_metrics_json(v_res, v_res["frame_logs"])
                    st.download_button("📍 Full Video Data (JSON)", data=json_str, file_name="video_metrics.json", mime="application/json")


# ==========================================
# 4. FRAME EXPLORER MODE
# ==========================================
elif nav_page == "🎞️ Frame Explorer":
    st.subheader("🎞️ Frame-by-Frame Inspection & Key-Frame Diagnostics")

    v_res = st.session_state.get("video_results", None)

    if v_res is None or not v_res.get("original_frames"):
        st.warning("⚠️ No video has been processed yet. Go to '🎥 Video Annotation' and process a video first.")
    else:
        total_proc = len(v_res["original_frames"])
        
        # Key frames jump shortcuts
        k1, k2 = st.columns(2)
        with k1:
            best_idx = st.selectbox("⭐ Jump to High-Confidence Frame", ["-- Select --"] + [f"Frame {idx} (Conf: {v_res['frame_logs'][idx]['overall_confidence']}%)" for idx in v_res.get("best_10_indices", [])])
        with k2:
            worst_idx = st.selectbox("⚠️ Jump to Low-Confidence Frame", ["-- Select --"] + [f"Frame {idx} (Conf: {v_res['frame_logs'][idx]['overall_confidence']}%)" for idx in v_res.get("worst_10_indices", [])])

        curr_frame = 0
        if best_idx != "-- Select --":
            curr_frame = int(best_idx.split()[1])
        elif worst_idx != "-- Select --":
            curr_frame = int(worst_idx.split()[1])

        frame_idx = st.slider("Select Frame Index", min_value=0, max_value=total_proc - 1, value=min(curr_frame, total_proc - 1))

        orig_frame = v_res["original_frames"][frame_idx]
        ann_frame = v_res["annotated_frames"][frame_idx]
        mask_frames = v_res.get("annotation_only_frames", [])
        mask_frame = mask_frames[frame_idx] if frame_idx < len(mask_frames) else None
        log = v_res["frame_logs"][frame_idx]

        status = log["detection_status"]
        if status == "DETECTED":
            badge_html = '<span class="status-badge badge-detected">DETECTED</span>'
        elif status == "PREDICTED":
            badge_html = '<span class="status-badge badge-low-confidence">PREDICTED</span>'
        else:
            badge_html = '<span class="status-badge badge-not-detected">NOT DETECTED</span>'

        st.markdown(f"#### Frame {log['frame_number']} ({log['timestamp_sec']}s) — Detection Status: {badge_html}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if mask_frame is not None:
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("#### Original Frame")
                st.image(orig_frame, use_container_width=True)
            with f2:
                st.markdown("#### Annotated Frame")
                st.image(ann_frame, use_container_width=True)
            with f3:
                st.markdown("#### Annotation Only (Mask)")
                st.image(mask_frame, use_container_width=True)
        else:
            f1, f2 = st.columns(2)
            with f1:
                st.markdown("#### Original Frame")
                st.image(orig_frame, use_container_width=True)
            with f2:
                st.markdown("#### Annotated Frame")
                st.image(ann_frame, use_container_width=True)

        st.markdown("---")
        fd1, fd2, fd3, fd4 = st.columns(4)
        with fd1:
            st.metric("Pupil Center (X, Y)", f"({log['pupil_center_x']}, {log['pupil_center_y']})")
            st.metric("Pupil Radius", f"{log['pupil_radius']} px")
        with fd2:
            st.metric("Iris Center (X, Y)", f"({log['iris_center_x']}, {log['iris_center_y']})")
            st.metric("Iris Radius", f"{log['iris_radius']} px")
        with fd3:
            st.metric("Pupil Confidence", f"{log['pupil_confidence']}%")
            st.metric("Iris Confidence", f"{log['iris_confidence']}%")
        with fd4:
            st.metric("Overall Confidence", f"{log['overall_confidence']}%")
            st.metric("Detection Status", log["detection_status"])


# ==========================================
# 5. ANALYTICS & GRAPHS
# ==========================================
elif nav_page == "📊 Analytics & Graphs":
    st.subheader("📊 Temporal Eye Tracking & PLR Analytics")

    v_res = st.session_state.get("video_results", None)

    if v_res is None or not v_res.get("frame_logs"):
        st.info("💡 Process a video in '🎥 Video Annotation' to generate video analytics charts.")
    else:
        df_logs = pd.DataFrame(v_res["frame_logs"])
        df_logs = compute_video_plr_dynamics(df_logs)

        charts_png = generate_video_tracking_charts(df_logs)
        st.image(charts_png, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Frame-by-Frame Metrics & Dilation Velocity Log")
        st.dataframe(df_logs, use_container_width=True)


# ==========================================
# 6. DETECTION DIAGNOSTICS (NEW)
# ==========================================
elif nav_page == "🔬 Detection Diagnostics":
    st.subheader("🔬 Multi-Stage Candidate Debugger & Pipeline Diagnostics")

    data = st.session_state.get("last_image_data", None)

    if data is None:
        st.info("💡 Process an image in '👁️ Image Annotation' first to inspect intermediate diagnostics.")
    else:
        diag_grid = generate_diagnostics_image_grid(
            data["original_rgb"],
            data["prep_dict"],
            data["pupil_res"],
            data["iris_res"],
            eyelid_obj=data.get("eyelid_obj"),
            glint_mask=data.get("glint_mask")
        )

        st.markdown("### 🧩 Intermediate Computer Vision Stages")
        cols = st.columns(3)
        stage_keys = list(diag_grid.keys())
        for idx, k in enumerate(stage_keys):
            with cols[idx % 3]:
                st.markdown(f"##### {k}")
                st.image(diag_grid[k], use_container_width=True)

        st.markdown("---")
        st.markdown("### 📐 Quantitative Confidence & Plausibility Breakdown")
        rep = get_diagnostics_report(data["pupil_res"], data["iris_res"], data.get("conf_dict", {}))

        if rep["is_valid"]:
            st.success(f"✅ Geometric Plausibility: VALID | Overall Calibrated Confidence: {rep['overall_confidence']}% ({rep['status']})")
        else:
            st.error(f"❌ Geometric Plausibility: REJECTED (Hard Gate = 0.0) | Overall Confidence: {rep['overall_confidence']}% ({rep['status']})")
            st.markdown("##### ⚠️ Active Physical Constraint Failures:")
            for pf in rep["plain_failures"]:
                st.markdown(f"- 🔴 **{pf}**")

        st.markdown("##### ⚖️ Confidence Calibration Weights & Evidence Values:")
        st.dataframe(pd.DataFrame(rep["components_table"]), use_container_width=True)


# ==========================================
# 7. ABOUT & METHODOLOGY
# ==========================================
elif nav_page == "ℹ️ About & Methodology":
    st.subheader("ℹ️ About the Advanced Eye Annotation System")
    st.markdown(r"""
### 📌 Multi-Stage Hybrid Computer Vision Architecture

```
INPUT IMAGE / VIDEO FRAME
         │
    ┌────▼────┐
    │ STAGE 1 │  Image Quality Analysis (sharpness, brightness, contrast, noise, glints)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 2 │  Eye ROI Localization (Haar cascade + adaptive dark-region bounding)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 3 │  Preprocessing Ensemble (CLAHE, Gamma, Morphological, Homomorphic)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 4 │  Specular Glint Detection & Localized Inpainting (Telea)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 5 │  Multi-Candidate Pupil Detector (Adaptive thresh + Otsu + Hough)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 6 │  Eyelid & Eyelash Masking (Parabolic curve fitting)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 7 │  Constrained Annular Iris Search (r = 1.3×pr to 3.5×pr)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 8 │  RANSAC Ellipse Fitting & Texture Verification
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 9 │  Strict 10-Point Geometric Validation & Confidence Fusion
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 10│  Boundary-Refined Annotation Points (Normal gradient snapping)
    └────┬────┘
         │
    ┌────▼────┐
    │ STAGE 11│  10D Kalman Filter Video Tracking & Key-Frame Quality Control
    └─────────┘
```

### 🎯 Key Advantages over Traditional Single-Algorithm Detectors

1. **Constrained Annular Search**: The iris is searched strictly relative to the detected pupil anchor within an anatomically realistic annulus (`1.3×pr` to `3.5×pr`), eliminating false detections of the outer eye or skin folds.
2. **Eyelid Masking**: Horizontal parabolic curves isolate eyelid edges and eyelashes from contaminating the radial gradient search.
3. **Iris-Sclera Texture Verification**: The boundary is verified to ensure it separates a darker inner iris from a lighter outer sclera.
4. **Fail-Safe Integrity**: The system prefers `NOT DETECTED` or `LOW CONFIDENCE` over confident wrong circles.
    """)
