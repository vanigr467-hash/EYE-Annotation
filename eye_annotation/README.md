# Automating Eye Annotation – Automatic Pupil and Iris Boundary Detection

An AI-based computer vision application designed to automatically detect and annotate pupil and iris boundaries in eye images, completely replacing manual point-by-point eye annotation.

---

## 📌 Problem Statement & Objective

In ophthalmic research, iris recognition systems, gaze tracking, and biometrics, human annotators manually mark dozens of points around the pupil and iris in every image. This manual process is time-consuming, expensive, and subject to human error.

**Project Objective**: Build a lightweight, CPU-efficient computer vision system that automatically:
1. Locates the pupil boundary and pupil center.
2. Locates the iris boundary and iris center.
3. Automatically generates $N$ evenly distributed boundary annotation points (32–128 points).
4. Calculates clinical biometric measurements (diameters, area ratios, decentration, eccentricity).
5. Provides an intuitive Streamlit web interface with export options (PNG, CSV, JSON, ZIP batch archive).

---

## ✨ Features

- **Automated Pupil Detection**: Multi-level thresholding, specular glint inpainting, contour circularity ranking, and circle/ellipse geometry fitting.
- **Automated Iris Boundary Detection**: Pupil-anchored radial integrodifferential gradient search (Daugman methodology) and Hough Circle fitting.
- **Parametric Point Generation**: Generates 16–128 customizable annotation points distributed evenly around pupil and iris boundaries.
- **Biometric & Diagnostic Analytics**:
  - Pupil & Iris Diameters & Areas
  - Iris-to-Pupil Diameter Ratio
  - Center Decentration $(\Delta X, \Delta Y, \text{Displacement Distance})$
  - Pupil Eccentricity / Ellipticity
  - Horizontal Intensity Profile Graph
- **Multi-Mode Streamlit Dashboard**:
  - 👁️ **Single Image Annotation Mode**: Interactive visualization, side-by-side comparison, parameter tuning.
  - 📁 **Batch Processing Mode**: Process multiple eye images concurrently and export summary report & ZIP package.
  - 🧪 **Synthetic Benchmark Lab**: Stress-test algorithm accuracy against procedurally generated eyes with adjustable glare and noise.
  - 📊 **Biometric Analytics**: View intensity cross-section plots.
  - 🎓 **How It Works**: Interactive visual explanation of the CV pipeline for project demonstrations.
- **Multi-Format Export Engine**: Download annotated PNG images, CSV metrics reports, JSON coordinate files, and batch ZIP packages.
- **Zero Paid APIs / GPU Required**: Runs 100% locally on standard CPU using open-source Python packages.

---

## 🛠️ Technology Stack

- **Python 3.10+**
- **Streamlit**: Web frontend and dashboard layout.
- **OpenCV (`opencv-python`)**: Computer vision operations, CLAHE, morphological filtering, contours, edge detection.
- **NumPy**: Matrix math and parametric point sampling.
- **Pillow**: Image conversion and saving.
- **Matplotlib**: Intensity profile plotting.
- **pandas**: Data handling and CSV export.
- **scipy / scikit-image**: Image analysis utilities.

---

## 📁 Project Structure

```
eye_annotation/
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Grayscale, Glint Inpainting, CLAHE, Bilateral filtering
│   ├── pupil_detection.py      # Pupil multi-candidate scoring, circle/ellipse fitting
│   ├── iris_detection.py       # Iris radial integrodifferential gradient search
│   ├── annotation.py           # Evenly spaced boundary point generator & custom overlays
│   ├── confidence.py           # Multi-factor confidence rating (High/Medium/Low)
│   ├── analytics.py            # Decentration, eccentricity & intensity profile math
│   └── visualization.py        # Plotting & preprocessing thumbnail grids
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Image conversion, PNG/CSV/JSON/ZIP exporters
│   └── synthetic_eye.py        # Procedural synthetic eye generator
├── sample_images/              # Pre-packaged sample eye images
└── outputs/                    # Default output directory
```

---

## 💻 Installation & Quick Start

### 1. Clone or Navigate to Directory
```bash
cd "c:\Users\vanig\GENERATIVE AI ASSIGNMENTS\EYE ANNOTATION"
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

**Activate Virtual Environment**:
- **Windows (PowerShell/CMD)**:
  ```powershell
  venv\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Application
```bash
streamlit run app.py
```

The application will launch automatically in your web browser at `http://localhost:8501`.

---

## 🔬 Computer Vision Methodology

```
Uploaded Eye Image ──► Specular Glint Inpainting ──► CLAHE Contrast Enhancement
                                                               │
   ┌───────────────────────────────────────────────────────────┴───────────────────────────────────────────┐
   ▼                                                                                                       ▼
Pupil Candidate Search (Multi-Threshold + Contour Circularity)                          Iris Search (Radial Gradient around Pupil Anchor)
   │                                                                                                       │
   ▼                                                                                                       ▼
Circle/Ellipse Fitting ─────────────────────────────────────────────────────────────► Parametric Boundary Point Generation (θ ∈ [0, 2π))
                                                                                                           │
                                                                                                           ▼
                                                                                           Visual Annotation & Export
```

1. **Specular Glint Inpainting**: High-intensity specular reflections inside the pupil are detected and inpainted using Telea inpainting to avoid breaking pupil contour circularity.
2. **CLAHE Enhancement**: Contrast-Limited Adaptive Histogram Equalization boosts contrast around the iris-sclera boundary without saturating skin tones.
3. **Pupil Candidate Scoring**: Candidates are evaluated using composite scoring: $S = 0.45 \times \text{circularity} + 0.45 \times \text{darkness} + 0.10 \times \text{centrality}$.
4. **Radial Integrodifferential Iris Search**: Evaluates intensity contrast steps along concentric arcs surrounding the detected pupil center to locate outer iris limbus.
5. **Boundary Point Sampling**: Distributes $N$ points along $x = cx + r\cos\theta, y = cy + r\sin\theta$.

---

## 🚀 Example Workflow

1. Open the application at `http://localhost:8501`.
2. Upload an eye image (JPG/PNG) or pick a sample image.
3. Adjust the **Annotation Points** slider (e.g. set to 32 or 64 points).
4. Review the automatically generated boundary lines and point markers.
5. Inspect pupil and iris metrics in the **Detection Results** panel.
6. Click **Download Annotated Image (PNG)** or **Download Metrics (CSV)**.

---

## 🔮 Limitations & Future Enhancements

- **Extreme Eyelid Occlusion**: Highly occluded eyes (eyes < 30% open) may require explicit eyelid contour segmentation.
- **Future Enhancements**: Integration of UNet deep learning segmentation models for complex pathological conditions.
