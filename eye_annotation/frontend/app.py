"""
Enterprise Eye Annotation Platform — Streamlit Frontend
Main entry point with multi-page navigation.
"""

import os
import sys
import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.components.api_client import api

# Page Configuration
st.set_page_config(
    page_title="Eye Annotation Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .status-badge { display: inline-block; padding: 0.3rem 0.75rem; border-radius: 16px; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; }
    .badge-approved { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .badge-pending { background: #FEF9C3; color: #A16207; border: 1px solid #FDE047; }
    .badge-rejected { background: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
    .badge-ai { background: #E0E7FF; color: #3730A3; border: 1px solid #A5B4FC; }
    .badge-human { background: #F0FDF4; color: #166534; border: 1px solid #86EFAC; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">👁️ EYE ANNOTATION PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industry-Grade Automated Pupil & Iris Boundary Detection with Human-in-the-Loop Review</div>', unsafe_allow_html=True)

# Backend connection check
backend_up = api.is_backend_running()
if backend_up:
    st.sidebar.success("🟢 Backend Connected")
else:
    st.sidebar.error("🔴 Backend Offline — Start with: `python run.py`")

# Navigation
st.sidebar.title("📌 Navigation")
nav_page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📁 Projects",
        "📤 Upload Media",
        "✏️ Annotate",
        "🤖 Auto-Detect",
        "📋 Review Queue",
        "🎥 Video Processing",
        "📥 Export",
        "🔬 Diagnostics",
        "ℹ️ About",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Enterprise Eye Annotation Platform v2.0")

# ============================================================
# DASHBOARD
# ============================================================
if nav_page == "🏠 Dashboard":
    st.subheader("📊 Platform Dashboard")

    if not backend_up:
        st.warning("⚠️ Backend is not running. Start the backend server first: `python run.py`")
        st.info("💡 The platform requires both the **FastAPI backend** (port 8000) and this **Streamlit frontend** (port 8501) running simultaneously.")
        st.code("python run.py", language="bash")
        st.stop()

    # Load stats
    try:
        projects = api.list_projects()
        media = api.list_media()
        images = [m for m in media if m.get("media_type") == "image"]
        videos = [m for m in media if m.get("media_type") == "video"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📁 Projects", len(projects))
        with c2:
            st.metric("🖼️ Images", len(images))
        with c3:
            st.metric("🎥 Videos", len(videos))
        with c4:
            try:
                queue = api.get_review_queue()
                st.metric("📋 Review Queue", len(queue))
            except Exception:
                st.metric("📋 Review Queue", "N/A")

        st.markdown("---")

        if projects:
            st.markdown("### Recent Projects")
            for p in projects[:5]:
                with st.expander(f"📁 {p['name']} (ID: {p['id']})"):
                    st.write(f"**Owner:** {p.get('owner', 'N/A')}")
                    st.write(f"**Description:** {p.get('description', 'No description')}")
                    st.write(f"**Created:** {p.get('created_at', 'N/A')}")
        else:
            st.info("No projects yet. Go to **📁 Projects** to create one.")

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

# ============================================================
# PROJECTS
# ============================================================
elif nav_page == "📁 Projects":
    st.subheader("📁 Project Management")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    # Create Project
    with st.expander("➕ Create New Project", expanded=False):
        with st.form("create_project"):
            name = st.text_input("Project Name", placeholder="e.g. Eye Dataset v1")
            desc = st.text_area("Description", placeholder="Brief description of this annotation project")
            owner = st.text_input("Owner", value="default")
            submitted = st.form_submit_button("Create Project")
            if submitted and name:
                try:
                    result = api.create_project(name, desc, owner)
                    st.success(f"✅ Project '{name}' created (ID: {result['id']})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # List Projects
    st.markdown("---")
    try:
        projects = api.list_projects()
        if projects:
            for p in projects:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**📁 {p['name']}** — {p.get('description', 'No description')}")
                with col2:
                    st.caption(f"Owner: {p.get('owner', 'N/A')} | Created: {p.get('created_at', '')[:10]}")
                with col3:
                    if st.button("🗑️ Delete", key=f"del_proj_{p['id']}"):
                        api.delete_project(p['id'])
                        st.rerun()
                st.divider()
        else:
            st.info("No projects yet. Create one above.")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================
# UPLOAD MEDIA
# ============================================================
elif nav_page == "📤 Upload Media":
    st.subheader("📤 Upload Images & Videos")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    # Select project
    try:
        projects = api.list_projects()
    except Exception:
        projects = []

    if not projects:
        st.warning("Create a project first in the **📁 Projects** tab.")
        st.stop()

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected = st.selectbox("Select Project", list(project_options.keys()))
    project_id = project_options[selected]

    st.markdown("---")
    tab_img, tab_vid = st.tabs(["🖼️ Upload Images", "🎥 Upload Video"])

    with tab_img:
        uploaded_files = st.file_uploader(
            "Upload Eye Images", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("📤 Upload All Images"):
            progress = st.progress(0.0)
            for i, f in enumerate(uploaded_files):
                try:
                    result = api.upload_media(project_id, f.read(), f.name)
                    st.success(f"✅ {f.name} uploaded (ID: {result['id']})")
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")
                progress.progress((i + 1) / len(uploaded_files))

    with tab_vid:
        uploaded_video = st.file_uploader("Upload Eye Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video and st.button("📤 Upload Video"):
            with st.spinner("Uploading video..."):
                try:
                    result = api.upload_media(project_id, uploaded_video.read(), uploaded_video.name)
                    st.success(f"✅ Video uploaded (ID: {result['id']})")

                    # Offer frame extraction
                    fps = st.number_input("Extract frames at FPS:", min_value=1.0, max_value=30.0, value=5.0)
                    if st.button("🎞️ Extract Frames"):
                        with st.spinner("Extracting frames..."):
                            extract_result = api.extract_frames(result['id'], fps=fps)
                            st.success(f"✅ Extracted {extract_result.get('frame_count', 'N/A')} frames")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Show existing media
    st.markdown("---")
    st.markdown("### 📂 Uploaded Media")
    try:
        media_list = api.list_media(project_id=project_id)
        if media_list:
            for m in media_list:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    icon = "🖼️" if m.get("media_type") == "image" else "🎥"
                    st.markdown(f"{icon} **{m.get('original_filename', 'Unknown')}**")
                with col2:
                    dims = f"{m.get('width', '?')}x{m.get('height', '?')}"
                    st.caption(f"Type: {m.get('media_type', '?')} | Size: {dims} | Status: {m.get('status', '?')}")
                with col3:
                    if st.button("🗑️", key=f"del_media_{m['id']}"):
                        api.delete_media(m['id'])
                        st.rerun()
        else:
            st.info("No media uploaded yet for this project.")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================
# ANNOTATE (Manual)
# ============================================================
elif nav_page == "✏️ Annotate":
    st.subheader("✏️ Manual Annotation Editor")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    from frontend.components.ellipse_editor import ellipse_editor_widget, render_annotation_preview
    import cv2
    import numpy as np

    try:
        projects = api.list_projects()
    except Exception:
        projects = []

    if not projects:
        st.warning("Create a project first.")
        st.stop()

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected = st.selectbox("Select Project", list(project_options.keys()), key="ann_proj")
    project_id = project_options[selected]

    # Get media for this project
    try:
        media_list = api.list_media(project_id=project_id)
    except Exception:
        media_list = []

    if not media_list:
        st.info("Upload media first in the **📤 Upload Media** tab.")
        st.stop()

    media_options = {f"{m.get('original_filename', 'Unknown')} (ID: {m['id']})": m['id'] for m in media_list}
    selected_media = st.selectbox("Select Media Asset", list(media_options.keys()))
    media_id = media_options[selected_media]

    # Get frames
    try:
        frames = api.list_frames(media_id)
    except Exception:
        frames = []

    if not frames:
        st.info("No frames available. For videos, extract frames first.")
        st.stop()

    # Frame selector
    if len(frames) > 1:
        frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
    else:
        frame_idx = 0

    frame = frames[frame_idx]
    frame_id = frame['id']

    # Load frame image
    try:
        img_bytes = api.get_frame_image_bytes(frame_id)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image_rgb is not None:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        else:
            st.error("Could not decode image")
            st.stop()
    except Exception as e:
        st.error(f"Error loading frame: {e}")
        st.stop()

    # Load existing annotations
    try:
        annotations = api.list_annotations(frame_id=frame_id)
    except Exception:
        annotations = []

    existing_pupil = None
    existing_iris = None
    for ann in annotations:
        params = {"cx": ann["cx"], "cy": ann["cy"], "rx": ann["rx"], "ry": ann["ry"], "angle": ann.get("angle", 0)}
        if ann["label_type"] == "pupil":
            existing_pupil = params
        elif ann["label_type"] == "iris":
            existing_iris = params

    # Editor columns
    col_edit, col_preview = st.columns([1, 1])

    with col_edit:
        st.markdown("### 🎛️ Ellipse Controls")

        edit_pupil = st.checkbox("Edit Pupil", value=True)
        edit_iris = st.checkbox("Edit Iris", value=True)

        pupil_params = None
        iris_params = None

        if edit_pupil:
            pupil_params = ellipse_editor_widget(
                image_rgb, "pupil", initial_params=existing_pupil, key_prefix="manual"
            )

        if edit_iris:
            iris_params = ellipse_editor_widget(
                image_rgb, "iris", initial_params=existing_iris, key_prefix="manual"
            )

        # Save button
        st.markdown("---")
        if st.button("💾 Save Annotations", type="primary"):
            saved = 0
            try:
                if pupil_params:
                    api.create_annotation(
                        label_type="pupil",
                        cx=pupil_params["cx"], cy=pupil_params["cy"],
                        rx=pupil_params["rx"], ry=pupil_params["ry"],
                        angle=pupil_params["angle"],
                        frame_id=frame_id, media_asset_id=media_id,
                        source="human", status="approved"
                    )
                    saved += 1

                if iris_params:
                    api.create_annotation(
                        label_type="iris",
                        cx=iris_params["cx"], cy=iris_params["cy"],
                        rx=iris_params["rx"], ry=iris_params["ry"],
                        angle=iris_params["angle"],
                        frame_id=frame_id, media_asset_id=media_id,
                        source="human", status="approved"
                    )
                    saved += 1

                st.success(f"✅ Saved {saved} annotation(s)")
            except Exception as e:
                st.error(f"Error saving: {e}")

    with col_preview:
        st.markdown("### 👁️ Preview")
        preview = render_annotation_preview(image_rgb, pupil_params, iris_params)
        st.image(preview, use_container_width=True)

        # Show annotation info
        if annotations:
            st.markdown("### 📝 Existing Annotations")
            for ann in annotations:
                status_class = "badge-approved" if ann["status"] == "approved" else "badge-pending" if ann["status"] == "pending_review" else "badge-rejected"
                source_class = "badge-ai" if ann["source"] == "ai" else "badge-human"
                st.markdown(
                    f'<span class="status-badge {status_class}">{ann["status"]}</span> '
                    f'<span class="status-badge {source_class}">{ann["source"]}</span> '
                    f'**{ann["label_type"].upper()}**: center=({ann["cx"]:.1f}, {ann["cy"]:.1f}) '
                    f'radii=({ann["rx"]:.1f}, {ann["ry"]:.1f})',
                    unsafe_allow_html=True
                )

# ============================================================
# AUTO-DETECT
# ============================================================
elif nav_page == "🤖 Auto-Detect":
    st.subheader("🤖 AI Auto-Detection")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    try:
        projects = api.list_projects()
    except Exception:
        projects = []

    if not projects:
        st.warning("Create a project first.")
        st.stop()

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected = st.selectbox("Select Project", list(project_options.keys()), key="detect_proj")
    project_id = project_options[selected]

    try:
        media_list = api.list_media(project_id=project_id)
    except Exception:
        media_list = []

    if not media_list:
        st.info("Upload media first.")
        st.stop()

    media_options = {f"{m.get('original_filename', 'Unknown')} (ID: {m['id']})": m['id'] for m in media_list}
    selected_media = st.selectbox("Select Media", list(media_options.keys()), key="detect_media")
    media_id = media_options[selected_media]

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 Run Auto-Detection (All Frames)", type="primary"):
            with st.spinner("Running AI detection on all frames..."):
                try:
                    result = api.detect_batch(media_id)
                    st.success(f"✅ Detection complete!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        # Single frame detection
        try:
            frames = api.list_frames(media_id)
        except Exception:
            frames = []

        if frames:
            frame_options = {f"Frame {f.get('frame_index', 0)} (ID: {f['id']})": f['id'] for f in frames}
            selected_frame = st.selectbox("Or detect single frame:", list(frame_options.keys()))
            frame_id = frame_options[selected_frame]

            if st.button("🔍 Detect This Frame"):
                with st.spinner("Running detection..."):
                    try:
                        result = api.detect_single_frame(frame_id)
                        st.success("✅ Detection complete!")
                        st.json(result)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============================================================
# REVIEW QUEUE
# ============================================================
elif nav_page == "📋 Review Queue":
    st.subheader("📋 Annotation Review Queue")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    from frontend.components.ellipse_editor import ellipse_editor_widget, render_annotation_preview
    import cv2
    import numpy as np

    # Batch approve controls
    with st.expander("⚡ Batch Operations"):
        threshold = st.slider("Auto-approve confidence threshold (%)", 0.0, 100.0, 85.0, 1.0)
        if st.button("✅ Batch Approve All Above Threshold"):
            try:
                result = api.batch_approve(threshold)
                st.success(f"✅ Batch approved {result.get('approved_count', 0)} annotations")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Load review queue
    try:
        queue = api.get_review_queue()
    except Exception as e:
        st.error(f"Error loading review queue: {e}")
        queue = []

    if not queue:
        st.success("🎉 Review queue is empty! All annotations have been reviewed.")
        st.stop()

    st.info(f"📋 **{len(queue)}** annotations pending review (sorted by confidence, lowest first)")

    # Review each annotation
    for i, ann in enumerate(queue[:20]):  # Show max 20 at a time
        with st.expander(
            f"#{ann['id']} — {ann['label_type'].upper()} | "
            f"Confidence: {ann.get('confidence', 0):.1f}% | "
            f"Source: {ann.get('source', '?')}",
            expanded=(i == 0)
        ):
            frame_id = ann.get("frame_id")
            if frame_id:
                try:
                    img_bytes = api.get_frame_image_bytes(frame_id)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    image_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image_rgb is not None:
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

                        # Show annotation preview
                        params = {"cx": ann["cx"], "cy": ann["cy"], "rx": ann["rx"], "ry": ann["ry"], "angle": ann.get("angle", 0)}
                        preview = render_annotation_preview(
                            image_rgb,
                            pupil_params=params if ann["label_type"] == "pupil" else None,
                            iris_params=params if ann["label_type"] == "iris" else None,
                        )

                        img_col, ctrl_col = st.columns([2, 1])
                        with img_col:
                            st.image(preview, use_container_width=True)
                        with ctrl_col:
                            st.markdown(f"**Type:** {ann['label_type'].upper()}")
                            st.markdown(f"**Center:** ({ann['cx']:.1f}, {ann['cy']:.1f})")
                            st.markdown(f"**Radii:** ({ann['rx']:.1f}, {ann['ry']:.1f})")
                            st.markdown(f"**Confidence:** {ann.get('confidence', 0):.1f}%")

                            b1, b2 = st.columns(2)
                            with b1:
                                if st.button("✅ Approve", key=f"approve_{ann['id']}"):
                                    try:
                                        api.approve_annotation(ann['id'])
                                        st.success("Approved!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))
                            with b2:
                                if st.button("❌ Reject", key=f"reject_{ann['id']}"):
                                    try:
                                        api.reject_annotation(ann['id'])
                                        st.warning("Rejected")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))

                except Exception as e:
                    st.error(f"Could not load frame image: {e}")
            else:
                st.warning("No frame associated with this annotation")

# ============================================================
# VIDEO PROCESSING
# ============================================================
elif nav_page == "🎥 Video Processing":
    st.subheader("🎥 Video Processing & Tracking")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    try:
        projects = api.list_projects()
    except Exception:
        projects = []

    if not projects:
        st.warning("Create a project first.")
        st.stop()

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected = st.selectbox("Select Project", list(project_options.keys()), key="vid_proj")
    project_id = project_options[selected]

    try:
        media_list = api.list_media(project_id=project_id, media_type="video")
    except Exception:
        media_list = []

    if not media_list:
        st.info("No videos uploaded. Go to **📤 Upload Media** to upload a video.")
        st.stop()

    media_options = {f"{m.get('original_filename', 'Unknown')} (ID: {m['id']})": m['id'] for m in media_list}
    selected_media = st.selectbox("Select Video", list(media_options.keys()), key="vid_media")
    media_id = media_options[selected_media]

    st.markdown("---")

    # Instant Video Annotation Card
    st.markdown("### ⚡ Instant Video Annotation (Real-Time 60+ FPS)")
    st.info("💡 Process and annotate your entire video in 1-2 seconds with zero frame extraction wait time.")

    v_set1, v_set2, v_set3, v_set4 = st.columns(4)
    with v_set1:
        vid_style = st.selectbox(
            "Boundary Representation",
            [
                "Circles / Ellipses",
                "Circles + Numbers",
                "Numbers Only",
                "Keypoints / Dots Only",
                "Full (Curves + Points + Numbers)"
            ],
            index=0,
            key=f"vid_style_{media_id}"
        )
    with v_set2:
        vid_show_vals = st.checkbox(
            "Show Values on Boundary",
            value=(vid_style in ["Circles + Numbers", "Numbers Only", "Full (Curves + Points + Numbers)"]),
            key=f"vid_show_vals_{media_id}"
        )
    with v_set3:
        vid_vals_fmt = st.selectbox(
            "Values Format",
            [
                "Radius & Center (r, x, y)",
                "Radius only (r)",
                "Full Geometry (cx, cy, rx, ry)"
            ],
            index=0,
            key=f"vid_vals_fmt_{media_id}"
        )
    with v_set4:
        vid_render_mode_choice = st.selectbox(
            "Video Output Format",
            [
                "Both (Annotated + Annotation-Only)",
                "Annotated Video Only",
                "Annotation-Only Video (Pure Mask)"
            ],
            index=0,
            key=f"vid_render_mode_{media_id}"
        )

    vid_render_mode_map = {
        "Both (Annotated + Annotation-Only)": "both",
        "Annotated Video Only": "annotated",
        "Annotation-Only Video (Pure Mask)": "annotation_only"
    }
    vid_render_mode = vid_render_mode_map.get(vid_render_mode_choice, "both")

    inst_btn_col1, inst_btn_col2 = st.columns(2)
    with inst_btn_col1:
        if st.button("⚡ Generate Annotated Video Immediately", type="primary", key="btn_instant_video"):
            try:
                media_item = api.get_media(media_id)
                from backend.services.storage import StorageService
                storage = StorageService()
                local_vid_path = storage.get_local_path(media_item["storage_url"])

                out_inst_path = os.path.join("storage", "exports", f"annotated_instant_{media_id}.mp4")
                os.makedirs(os.path.dirname(out_inst_path), exist_ok=True)

                with st.spinner("⚡ Running ultra-fast anchor-locked tracking (60+ FPS)..."):
                    from src.fast_video_annotator import process_eye_video_instant
                    res_inst = process_eye_video_instant(
                        local_vid_path,
                        out_inst_path,
                        render_mode=vid_render_mode,
                        annotation_style=vid_style,
                        show_values=vid_show_vals,
                        values_format=vid_vals_fmt
                    )
                    st.session_state[f"instant_res_{media_id}"] = res_inst
                    st.success(f"✅ Done in {res_inst['processing_time_sec']} seconds ({res_inst['processing_fps']} FPS)!")
            except Exception as e:
                st.error(f"Error: {e}")

    inst_res_data = st.session_state.get(f"instant_res_{media_id}")
    if inst_res_data:
        ann_path = inst_res_data.get("annotated_video_path")
        mask_path = inst_res_data.get("annotation_only_video_path")

        views = []
        if ann_path and os.path.exists(ann_path):
            views.append("👁️ Annotated Video (Original + Overlay)")
        if mask_path and os.path.exists(mask_path):
            views.append("🖤 Annotation-Only Video (Pure Mask)")

        if views:
            if len(views) > 1:
                chosen_view = st.radio("Select Video Preview:", views, horizontal=True, key=f"preview_mode_{media_id}")
            else:
                chosen_view = views[0]

            if chosen_view.startswith("👁️") and ann_path and os.path.exists(ann_path):
                st.video(ann_path)
            elif mask_path and os.path.exists(mask_path):
                st.video(mask_path)
        elif os.path.exists(inst_res_data.get("output_video_path", "")):
            st.video(inst_res_data["output_video_path"])

        dl_cols = st.columns(3)
        c_i = 0
        if ann_path and os.path.exists(ann_path):
            with dl_cols[c_i]:
                with open(ann_path, "rb") as vf:
                    st.download_button(
                        "📥 Download Annotated Video (MP4)",
                        data=vf.read(),
                        file_name=f"annotated_video_{media_id}.mp4",
                        mime="video/mp4",
                        key=f"dl_ann_{media_id}"
                    )
            c_i += 1
        if mask_path and os.path.exists(mask_path):
            with dl_cols[c_i]:
                with open(mask_path, "rb") as vf:
                    st.download_button(
                        "🖤 Download Annotation-Only Video (MP4)",
                        data=vf.read(),
                        file_name=f"annotation_only_{media_id}.mp4",
                        mime="video/mp4",
                        key=f"dl_mask_{media_id}"
                    )
            c_i += 1
        with dl_cols[c_i]:
            st.metric("Processing Speed", f"{inst_res_data['processing_fps']} FPS", delta=f"{inst_res_data['processing_time_sec']}s total")

    st.markdown("---")
    st.markdown("### 🎞️ Advanced Frame-by-Frame Extraction (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        fps = st.number_input("Frame Extraction FPS", min_value=1.0, max_value=30.0, value=5.0, key="vid_fps")
        if st.button("🎞️ Extract Frames"):
            with st.spinner("Extracting video frames..."):
                try:
                    result = api.extract_frames(media_id, fps=fps)
                    st.success(f"✅ Extracted {result.get('frame_count', '?')} frames")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("🤖 Run Detection on All Frames", type="primary"):
            with st.spinner("Running AI detection on all video frames..."):
                try:
                    result = api.detect_batch(media_id)
                    st.success("✅ Detection complete!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Frame browser
    st.markdown("---")
    st.markdown("### 🎞️ Frame Browser")
    try:
        frames = api.list_frames(media_id)
    except Exception:
        frames = []

    if frames:
        import cv2
        import numpy as np
        from frontend.components.ellipse_editor import render_annotation_preview

        frame_idx = st.slider("Frame Index", 0, len(frames) - 1, 0, key="vid_frame_slider")
        frame = frames[frame_idx]
        frame_id = frame['id']

        try:
            img_bytes = api.get_frame_image_bytes(frame_id)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image_rgb is not None:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

                # Get annotations for this frame
                anns = api.list_annotations(frame_id=frame_id)
                pupil_p = iris_p = None
                for a in anns:
                    p = {"cx": a["cx"], "cy": a["cy"], "rx": a["rx"], "ry": a["ry"], "angle": a.get("angle", 0)}
                    if a["label_type"] == "pupil":
                        pupil_p = p
                    elif a["label_type"] == "iris":
                        iris_p = p

                fc1, fc2 = st.columns(2)
                with fc1:
                    st.image(image_rgb, caption=f"Frame {frame.get('frame_index', 0)} (Original)", use_container_width=True)
                with fc2:
                    preview = render_annotation_preview(image_rgb, pupil_p, iris_p)
                    st.image(preview, caption=f"Frame {frame.get('frame_index', 0)} (Annotated)", use_container_width=True)

                # Show annotation details
                if anns:
                    for a in anns:
                        badge = "badge-approved" if a["status"] == "approved" else "badge-pending"
                        st.markdown(
                            f'<span class="status-badge {badge}">{a["status"]}</span> '
                            f'**{a["label_type"].upper()}** — center=({a["cx"]:.1f}, {a["cy"]:.1f}) '
                            f'radii=({a["rx"]:.1f}, {a["ry"]:.1f}) conf={a.get("confidence", 0):.1f}%',
                            unsafe_allow_html=True
                        )
        except Exception as e:
            st.error(f"Error loading frame: {e}")
    else:
        st.info("No frames extracted yet. Use the controls above to extract frames.")

# ============================================================
# EXPORT
# ============================================================
elif nav_page == "📥 Export":
    st.subheader("📥 Export Datasets")

    if not backend_up:
        st.warning("⚠️ Backend offline")
        st.stop()

    try:
        projects = api.list_projects()
    except Exception:
        projects = []

    if not projects:
        st.warning("Create a project first.")
        st.stop()

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected = st.selectbox("Select Project", list(project_options.keys()), key="export_proj")
    project_id = project_options[selected]

    st.markdown("---")

    import json

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📦 COCO JSON"):
            try:
                data = api.export_coco(project_id)
                st.download_button(
                    "📥 Download COCO JSON",
                    data=json.dumps(data, indent=2),
                    file_name="coco_annotations.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("📊 CSV"):
            try:
                data = api.export_csv(project_id)
                st.download_button(
                    "📥 Download CSV",
                    data=data,
                    file_name="annotations.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error: {e}")

    with col3:
        if st.button("⚡ YOLO"):
            try:
                data = api.export_yolo(project_id)
                st.download_button(
                    "📥 Download YOLO",
                    data=data,
                    file_name="yolo_annotations.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {e}")

    with col4:
        if st.button("📈 Summary"):
            try:
                data = api.export_summary(project_id)
                st.json(data)
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# DIAGNOSTICS (Uses legacy pipeline directly)
# ============================================================
elif nav_page == "🔬 Diagnostics":
    st.subheader("🔬 CV Pipeline Diagnostics")
    st.info("💡 This page runs the CV detection pipeline directly (without the API) for debugging purposes.")

    import cv2
    import numpy as np
    from PIL import Image

    try:
        from src.preprocessing import preprocess_eye_image
        from src.pupil_detector import detect_pupil
        from src.iris_detector import detect_iris
        from src.eyelid_mask import detect_eyelid_mask
        from src.annotation import draw_automatic_annotations, draw_annotation_only_mask
        from src.confidence import compute_overall_confidence
        from src.validation import validate_detection_pair
        from src.diagnostics import generate_diagnostics_image_grid, get_diagnostics_report
        from utils.synthetic_eye import generate_synthetic_eye

        d_opt1, d_opt2, d_opt3 = st.columns(3)
        with d_opt1:
            diag_style = st.selectbox(
                "Boundary Representation",
                [
                    "Circles / Ellipses",
                    "Circles + Numbers",
                    "Numbers Only",
                    "Keypoints / Dots Only",
                    "Full (Curves + Points + Numbers)"
                ],
                index=0,
                key="diag_style_opt"
            )
        with d_opt2:
            diag_show_vals = st.checkbox(
                "Show Values on Boundary",
                value=(diag_style in ["Circles + Numbers", "Numbers Only", "Full (Curves + Points + Numbers)"]),
                key="diag_show_vals_opt"
            )
        with d_opt3:
            diag_vals_fmt = st.selectbox(
                "Values Format",
                [
                    "Radius & Center (r, x, y)",
                    "Radius only (r)",
                    "Full Geometry (cx, cy, rx, ry)"
                ],
                index=0,
                key="diag_vals_fmt_opt"
            )

        uploaded = st.file_uploader("Upload eye image for diagnostics", type=["jpg", "jpeg", "png"])

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_rgb, _ = generate_synthetic_eye(pupil_offset=(8, -5))
            st.info("Using synthetic eye image. Upload a real image for accurate diagnostics.")

        h, w = image_rgb.shape[:2]
        prep_dict = preprocess_eye_image(image_rgb, clahe_clip=2.0, glint_removal=True)
        pupil_res = detect_pupil(prep_dict, use_starburst_refinement=True)

        eyelid_obj = None
        eyelid_mask = None
        if pupil_res.detected:
            eyelid_obj = detect_eyelid_mask(prep_dict["gray_inpainted"], pupil_res.center, pupil_res.radius)
            eyelid_mask = eyelid_obj.valid_mask

        iris_res = detect_iris(prep_dict, pupil_res, eyelid_mask=eyelid_mask)

        annotated_rgb, points_data = draw_automatic_annotations(
            image_rgb, pupil_res, iris_res, num_points=64,
            show_boundary_evidence=True, eyelid_mask_obj=eyelid_obj,
            boundary_style=diag_style,
            show_values=diag_show_vals,
            values_format=diag_vals_fmt
        )

        mask_rgb = draw_annotation_only_mask(
            image_rgb.shape, pupil_res, iris_res, num_points=64,
            boundary_style=diag_style,
            show_values=diag_show_vals,
            values_format=diag_vals_fmt,
            eyelid_mask_obj=eyelid_obj
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(image_rgb, caption="Original", use_container_width=True)
        with c2:
            diag_view = st.radio("Display View", ["👁️ Annotated", "🖤 Annotation-Only (Mask)"], horizontal=True, key="diag_view_mode")
            if diag_view.startswith("👁️"):
                st.image(annotated_rgb, caption="Detected Overlay", use_container_width=True)
            else:
                st.image(mask_rgb, caption="Pure Annotation Mask", use_container_width=True)

        # Diagnostics grid
        diag_grid = generate_diagnostics_image_grid(
            image_rgb, prep_dict, pupil_res, iris_res,
            eyelid_obj=eyelid_obj, glint_mask=prep_dict.get("glint_mask")
        )

        st.markdown("### 🧩 Pipeline Stages")
        cols = st.columns(3)
        for idx, (k, v) in enumerate(diag_grid.items()):
            with cols[idx % 3]:
                st.markdown(f"##### {k}")
                st.image(v, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📐 Quantitative Confidence & Plausibility Breakdown")
        conf_dict = compute_overall_confidence(pupil_res, iris_res, frame_shape=(h, w))
        rep = get_diagnostics_report(pupil_res, iris_res, conf_dict)

        if rep["is_valid"]:
            st.success(f"✅ Geometric Plausibility: VALID | Overall Calibrated Confidence: {rep['overall_confidence']}% ({rep['status']})")
        else:
            st.error(f"❌ Geometric Plausibility: REJECTED (Hard Gate = 0.0) | Overall Confidence: {rep['overall_confidence']}% ({rep['status']})")
            st.markdown("##### ⚠️ Active Physical Constraint Failures:")
            for pf in rep["plain_failures"]:
                st.markdown(f"- 🔴 **{pf}**")

        st.markdown("##### ⚖️ Confidence Calibration Weights & Evidence Values:")
        import pandas as pd
        st.dataframe(pd.DataFrame(rep["components_table"]), use_container_width=True)

    except ImportError as e:
        st.error(f"CV modules not available: {e}")

# ============================================================
# ABOUT
# ============================================================
elif nav_page == "ℹ️ About":
    st.subheader("ℹ️ About the Eye Annotation Platform")
    st.markdown(r"""
### 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Streamlit UI   │────▶│  FastAPI Backend  │────▶│  SQLite Database │
│ (Annotate,      │◀────│  (REST API)      │◀────│  (Metadata,      │
│  Review,        │     │                  │     │   Annotations)   │
│  Dashboard)     │     └────────┬─────────┘     └──────────────────┘
└─────────────────┘              │
                                 ▼
                    ┌──────────────────────┐     ┌──────────────────┐
                    │  ML Inference Module │     │  Local Storage   │
                    │  (Swirski Pupil,     │     │  (Images, Frames │
                    │   Daugman Iris,      │     │   Masks, Exports)│
                    │   KLT Tracker)       │     └──────────────────┘
                    └──────────────────────┘
```

### 📌 Key Features

1. **Project Management**: Organize annotations by project
2. **Media Ingest**: Upload images and videos, extract video frames
3. **AI Auto-Detection**: Swirski pupil + Daugman iris detection pipeline
4. **Manual Annotation**: Interactive ellipse editor with real-time preview
5. **Human-in-the-Loop Review**: Approve, reject, or edit AI predictions
6. **Confidence Scoring**: Auto-route low-confidence annotations to review
7. **Audit Trail**: Full version history of all annotation edits
8. **Dataset Export**: COCO JSON, CSV, YOLO keypoints formats
9. **Video Tracking**: KLT optical flow + Kalman filter temporal tracking

### 🔬 CV Pipeline (11 Stages)

| Stage | Description |
|---|---|
| 1 | Image Quality Analysis (sharpness, brightness, noise) |
| 2 | Eye ROI Localization (Haar cascade + dark-region) |
| 3 | Preprocessing Ensemble (CLAHE, Gamma, Morphological) |
| 4 | Specular Glint Detection & Inpainting |
| 5 | Multi-Candidate Pupil Detection (Swirski Center-Surround) |
| 6 | Eyelid & Eyelash Masking |
| 7 | Constrained Annular Iris Search (Daugman IDO) |
| 8 | RANSAC Fitzgibbon Ellipse Fitting |
| 9 | Geometric Validation & Confidence Fusion |
| 10 | Anti-Aliased Boundary Point Generation |
| 11 | 10D Kalman Filter Video Tracking (KLT Optical Flow) |

### 📊 Data Model

| Entity | Purpose |
|---|---|
| **Project** | Groups related media and annotations |
| **MediaAsset** | An uploaded image or video |
| **Frame** | Individual frame (from image or extracted from video) |
| **Annotation** | Ellipse boundary (pupil/iris) with source, confidence, status |
| **AnnotationVersion** | Audit trail of edits |
| **User** | Platform user with role-based access |
    """)
