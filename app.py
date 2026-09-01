import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_eye_image
from src.pupil_detection import detect_pupil
from src.iris_detection import detect_iris
from src.annotation import draw_automatic_annotations, COLOR_SCHEMES
from src.confidence import compute_overall_confidence
from src.analytics import compute_eye_analytics
from src.visualization import generate_intensity_profile_plot, create_preprocessing_thumbnail_grid
from utils.image_utils import (
    load_image_from_bytes, numpy_to_png_bytes, resize_if_large,
    export_results_to_csv, export_points_to_json, create_batch_zip
)
from utils.synthetic_eye import generate_synthetic_eye

# Page Configuration
st.set_page_config(
    page_title="Automated Eye Annotation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .confidence-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .confidence-med {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .confidence-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">👁️ Automated Eye Annotation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-assisted automatic detection of pupil and iris boundaries without manual point-by-point annotation</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Controls & Parameters")

# Navigation / Mode Tabs
app_mode = st.sidebar.radio(
    "Select Operating Mode",
    ["👁️ Single Image Mode", "📁 Batch Processing Mode", "🧪 Synthetic Benchmark Lab", "📊 Biometric Analytics", "🎓 How It Works"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Detection Parameters")

fit_mode = st.sidebar.selectbox("Fitting Geometry", ["circle", "ellipse"], index=0, help="Select circle or ellipse fitting for pupil boundary.")
num_points = st.sidebar.slider("Annotation Boundary Points", min_value=16, max_value=128, value=32, step=8, help="Number of points generated around pupil/iris.")
sensitivity = st.sidebar.slider("Detection Sensitivity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
clahe_clip = st.sidebar.slider("CLAHE Contrast Clip Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
glint_removal = st.sidebar.checkbox("Specular Glint Inpainting", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visual Overlay Settings")
theme = st.sidebar.selectbox("Color Theme", list(COLOR_SCHEMES.keys()), index=0)
show_pupil_layer = st.sidebar.checkbox("Show Pupil Boundary", value=True)
show_iris_layer = st.sidebar.checkbox("Show Iris Boundary", value=True)
show_centers_layer = st.sidebar.checkbox("Show Center Crosshairs", value=True)
show_points_layer = st.sidebar.checkbox("Show Boundary Points", value=True)
show_labels_layer = st.sidebar.checkbox("Show Point Labels", value=False)
show_prep_checkbox = st.sidebar.checkbox("Show Preprocessing Pipeline", value=False)


# Helper function to process a single image array
def process_single_eye_image(img_rgb: np.ndarray):
    resized_img, scale = resize_if_large(img_rgb, max_dim=1000)
    prep_dict = preprocess_eye_image(
        resized_img,
        clahe_clip=clahe_clip,
        blur_kernel=5,
        glint_removal=glint_removal
    )
    pupil_res = detect_pupil(prep_dict, sensitivity=sensitivity, fit_mode=fit_mode)
    iris_res = detect_iris(prep_dict, pupil_res, sensitivity=sensitivity, fit_mode=fit_mode)
    
    annotated_rgb, points_data = draw_automatic_annotations(
        resized_img,
        pupil_res,
        iris_res,
        num_points=num_points,
        theme=theme,
        show_pupil=show_pupil_layer,
        show_iris=show_iris_layer,
        show_centers=show_centers_layer,
        show_points=show_points_layer,
        show_labels=show_labels_layer
    )
    
    conf_dict = compute_overall_confidence(pupil_res, iris_res)
    analytics_dict = compute_eye_analytics(prep_dict, pupil_res, iris_res)
    
    return {
        "resized_rgb": resized_img,
        "prep_dict": prep_dict,
        "pupil_res": pupil_res,
        "iris_res": iris_res,
        "annotated_rgb": annotated_rgb,
        "points_data": points_data,
        "confidence": conf_dict,
        "analytics": analytics_dict
    }


# ==========================================
# MODE 1: SINGLE IMAGE ANNOTATION
# ==========================================
if app_mode == "👁️ Single Image Mode":
    st.subheader("1. Select or Upload Eye Image")
    
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        uploaded_file = st.file_uploader("Upload an Eye Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        
    with upload_col2:
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".png"))] if os.path.exists(sample_dir) else []
        selected_sample = st.selectbox("Or choose a pre-loaded sample", ["-- None --"] + sample_files)

    image_bytes = None
    image_name = "eye_image.png"

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_name = uploaded_file.name
    elif selected_sample != "-- None --":
        sample_path = os.path.join(sample_dir, selected_sample)
        with open(sample_path, "rb") as f:
            image_bytes = f.read()
        image_name = selected_sample

    if image_bytes is None:
        # Generate default demonstration synthetic eye if no upload
        default_rgb, _ = generate_synthetic_eye(width=600, height=450, pupil_r=45, iris_r=120)
        img_input = default_rgb
        st.info("💡 Showing default synthetic eye image. Upload a custom image or pick a sample from above.")
    else:
        try:
            img_input = load_image_from_bytes(image_bytes)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

    # Process Image
    results = process_single_eye_image(img_input)
    
    # Display Original vs Annotated Comparison
    st.markdown("---")
    st.subheader("2. Automatic Boundary Detection & Annotation")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("#### 📷 Original Image")
        st.image(results["resized_rgb"], use_container_width=True)
        h_orig, w_orig = results["resized_rgb"].shape[:2]
        st.caption(f"Resolution: {w_orig} x {h_orig} px | Channels: 3 (RGB)")
        
    with comp_col2:
        st.markdown(f"#### 🎯 Automatically Annotated Image ({num_points} Points)")
        st.image(results["annotated_rgb"], use_container_width=True)
        st.caption("Pupil & Iris boundaries with automatically generated annotation markers")

    # Metrics Section
    st.markdown("---")
    st.subheader("3. Detection Results & Biometric Measurements")
    
    an = results["analytics"]
    cf = results["confidence"]
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.metric("Pupil Center (X, Y)", f"({an['pupil_center_x']}, {an['pupil_center_y']})")
        st.metric("Pupil Radius", f"{an['pupil_radius_px']} px")
        
    with m_col2:
        st.metric("Pupil Diameter", f"{an['pupil_diameter_px']} px")
        st.metric("Pupil Area", f"{an['pupil_area_px2']} px²")
        
    with m_col3:
        st.metric("Iris Center (X, Y)", f"({an['iris_center_x']}, {an['iris_center_y']})")
        st.metric("Iris Radius", f"{an['iris_radius_px']} px")
        
    with m_col4:
        st.metric("Iris Diameter", f"{an['iris_diameter_px']} px")
        st.metric("Iris-to-Pupil Ratio", f"{an['iris_pupil_diameter_ratio']} x")

    with m_col5:
        st.metric("Decentration (Δ Dist)", f"{an['center_decentration_distance']} px")
        st.metric("Pupil Eccentricity", f"{an['pupil_eccentricity']}")

    # Confidence Rating Card
    st.markdown("#### 🛡️ Detection Confidence Rating")
    conf_level = cf["overall_level"]
    if conf_level == "High":
        st.markdown(f'<span class="confidence-high">High Confidence ({cf["overall_confidence"]}%)</span>', unsafe_allow_html=True)
    elif conf_level == "Medium":
        st.markdown(f'<span class="confidence-med">Medium Confidence ({cf["overall_confidence"]}%)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="confidence-low">Low Confidence ({cf["overall_confidence"]}%)</span>', unsafe_allow_html=True)

    st.write(cf["warning"])

    # Download Results Buttons
    st.markdown("---")
    st.subheader("4. Download Results")

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    annotated_png = numpy_to_png_bytes(results["annotated_rgb"])
    results_csv = export_results_to_csv(an)
    points_json = export_points_to_json(results["points_data"])

    with dl_col1:
        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=annotated_png,
            file_name=f"annotated_{image_name}",
            mime="image/png"
        )

    with dl_col2:
        st.download_button(
            label="📊 Download Metrics (CSV)",
            data=results_csv,
            file_name="eye_metrics_report.csv",
            mime="text/csv"
        )

    with dl_col3:
        st.download_button(
            label="📍 Download Point Coordinates (JSON)",
            data=points_json,
            file_name="boundary_points.json",
            mime="application/json"
        )

    # Optional Preprocessing View
    if show_prep_checkbox:
        st.markdown("---")
        st.subheader("🔍 Processing Details & CV Pipeline Stages")
        grid_dict = create_preprocessing_thumbnail_grid(results["prep_dict"])
        p_cols = st.columns(len(grid_dict))
        for idx, (title, img_arr) in enumerate(grid_dict.items()):
            with p_cols[idx]:
                st.markdown(f"**{title}**")
                st.image(img_arr, use_container_width=True)


# ==========================================
# MODE 2: BATCH PROCESSING MODE
# ==========================================
elif app_mode == "📁 Batch Processing Mode":
    st.subheader("📁 Batch Image Processing Mode")
    st.write("Upload multiple eye images to process and annotate them automatically in batch mode.")
    
    batch_files = st.file_uploader(
        "Upload Multiple Eye Images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if batch_files:
        st.info(f"Loaded {len(batch_files)} images for batch processing.")
        if st.button("🚀 Start Batch Annotation"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            
            for idx, bfile in enumerate(batch_files):
                status_text.text(f"Processing image {idx+1}/{len(batch_files)}: {bfile.name}")
                file_bytes = bfile.read()
                img_rgb = load_image_from_bytes(file_bytes)
                
                res = process_single_eye_image(img_rgb)
                
                batch_results.append({
                    "filename": bfile.name,
                    "image_rgb": res["annotated_rgb"],
                    "metrics": res["analytics"]
                })
                
                progress_bar.progress((idx + 1) / len(batch_files))
                
            status_text.text("✅ Batch processing complete!")
            
            # Create summary dataframe
            summary_metrics = [r["metrics"] for r in batch_results]
            for idx, r in enumerate(summary_metrics):
                r["filename"] = batch_results[idx]["filename"]
                
            df_summary = pd.DataFrame(summary_metrics)
            st.dataframe(df_summary, use_container_width=True)
            
            # Zip download
            zip_bytes = create_batch_zip(batch_results)
            st.download_button(
                label="📦 Download Batch Results (ZIP)",
                data=zip_bytes,
                file_name="batch_eye_annotations.zip",
                mime="application/zip"
            )


# ==========================================
# MODE 3: SYNTHETIC BENCHMARK LAB
# ==========================================
elif app_mode == "🧪 Synthetic Benchmark Lab":
    st.subheader("🧪 Interactive Synthetic Eye Benchmark Lab")
    st.write("Stress-test the computer vision detection algorithm by generating synthetic eyes with custom parameters.")
    
    b_col1, b_col2 = st.columns([1, 2])
    
    with b_col1:
        st.markdown("#### Synthetic Image Controls")
        syn_pr = st.slider("Ground Truth Pupil Radius", 20, 80, 45)
        syn_ir = st.slider("Ground Truth Iris Radius", 80, 180, 120)
        syn_off_x = st.slider("Pupil Offset X", -30, 30, 0)
        syn_off_y = st.slider("Pupil Offset Y", -30, 30, 0)
        syn_glint = st.checkbox("Include Specular Glint", value=True)
        syn_noise = st.slider("Noise Level", 0.0, 0.15, 0.03, step=0.01)
        syn_blur = st.slider("Blur Kernel Level", 1, 9, 1, step=2)

    with b_col2:
        syn_rgb, gt = generate_synthetic_eye(
            width=600, height=450,
            pupil_r=syn_pr, iris_r=syn_ir,
            pupil_offset=(syn_off_x, syn_off_y),
            glint=syn_glint, noise_level=syn_noise, blur_level=syn_blur
        )
        
        bench_res = process_single_eye_image(syn_rgb)
        
        st.markdown("#### Benchmark Evaluation")
        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            st.image(syn_rgb, caption="Synthetic Ground Truth Image", use_container_width=True)
        with eval_col2:
            st.image(bench_res["annotated_rgb"], caption="Algorithm Detected Output", use_container_width=True)

        # Accuracy comparison table
        gt_px, gt_py = gt["pupil_center"]
        det_px, det_py = bench_res["analytics"]["pupil_center_x"], bench_res["analytics"]["pupil_center_y"]
        p_error = np.sqrt((gt_px - det_px)**2 + (gt_py - det_py)**2)
        
        st.markdown("#### Ground Truth vs Detected Accuracy")
        st.write(f"• Pupil Center Error: **{p_error:.2f} px**")
        st.write(f"• Pupil Radius Error: **{abs(gt['pupil_radius'] - bench_res['analytics']['pupil_radius_px']):.2f} px**")
        st.write(f"• Iris Radius Error: **{abs(gt['iris_radius'] - bench_res['analytics']['iris_radius_px']):.2f} px**")


# ==========================================
# MODE 4: BIOMETRIC ANALYTICS
# ==========================================
elif app_mode == "📊 Biometric Analytics":
    st.subheader("📊 Biometric Intensity Profile & Radial Cross-Section")
    st.write("Analyzes the brightness profile across the horizontal center line of the pupil and iris.")
    
    # Process default or sample image
    default_rgb, _ = generate_synthetic_eye(width=600, height=450, pupil_r=45, iris_r=120)
    res = process_single_eye_image(default_rgb)
    an = res["analytics"]
    
    profile_png = generate_intensity_profile_plot(
        an["profile_x"],
        an["profile_intensity"],
        an["pupil_center_x"],
        an["pupil_radius_px"],
        an["iris_center_x"],
        an["iris_radius_px"]
    )
    
    st.image(profile_png, use_container_width=True)
    
    st.markdown("#### Diagnostic Insights")
    st.write("""
    - **Pupil Region**: Corresponds to the deepest intensity trough (darkest gray values near 0-40).
    - **Iris Region**: Intermediate intensity band (values around 60-120) with characteristic texture fluctuations.
    - **Sclera Boundary**: Sharp intensity gradient transition upwards to high values (200-245).
    """)


# ==========================================
# MODE 5: HOW IT WORKS & DEMONSTRATION
# ==========================================
elif app_mode == "🎓 How It Works":
    st.subheader("🎓 How The Automated Eye Annotation System Works")
    st.markdown("""
    This project automates eye annotation using lightweight, CPU-efficient computer vision algorithms. 
    It eliminates the need for human operators to manually click points around pupils and irises.

    ---

    ### 🔄 Computer Vision Pipeline Breakdown

    ```
    ┌─────────────────────────┐
    │ 1. Uploaded Eye Image   │
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 2. Preprocessing        │ ── Grayscale, Glint Inpainting, CLAHE Contrast Enhancement
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 3. Pupil Detection      │ ── Multi-Thresholding, Contour Circularity & Darkness Scoring
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 4. Iris Detection       │ ── Pupil Anchor Reference, Radial Integrodifferential Gradient Search
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 5. Point Generation     │ ── Sample N evenly spaced points along circumference θ ∈ [0, 2π)
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 6. Output & Export      │ ── Visual Overlay, Confidence Badge, PNG / CSV / JSON Downloads
    └─────────────────────────┘
    ```

    ---

    ### 🛠️ Key Computer Vision Algorithms Used
    - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast between iris and sclera without blowing out glint reflections.
    - **Specular Reflection Inpainting**: Detects bright glint spots inside the pupil and inpaints them before contour analysis to prevent boundary distortion.
    - **Contour Circularity & Candidate Ranking**: Evaluates shape circularity $\\frac{4\\pi A}{P^2}$ and darkness to select true pupil candidates without hardcoded assumptions.
    - **Daugman-Inspired Radial Integrodifferential Operator**: Computes radial intensity gradient steps along concentric arcs around the pupil center to locate outer iris boundary (limbus).
    - **Parametric Circumference Sampling**: Evenly distributes $N$ boundary points along $x = cx + r\\cos\\theta, y = cy + r\\sin\\theta$.
    """)
