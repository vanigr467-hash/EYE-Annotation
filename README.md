# Advanced Automated Eye Annotation & Gaze Tracking System

An advanced computer-vision system designed to automatically detect, annotate, and temporally track pupil and iris boundaries in static images and video sequences with sub-pixel Starburst RANSAC ray-tracing, 2D Eye Gaze Vectors, Pupillary Light Reflex (PLR) dynamics, and ML dataset exports (COCO/YOLO keypoints).

---

## 📌 Problem Statement & Objective

Manual eye annotation requires human operators to manually mark dozens of points around the pupil and iris across thousands of frames. This manual process is time-consuming, expensive, and subject to human error and inter-frame jitter.

**Project Objective**: Build an advanced computer vision system that:
1. Automatically detects pupil and iris boundaries in images and videos with sub-pixel Starburst RANSAC ray-tracing refinement.
2. Calculates 2D Eye Gaze Direction Vectors $\vec{G} = (X_{pupil} - X_{iris}, Y_{pupil} - Y_{iris})$, Gaze Angles, and renders visual Gaze Direction Arrows.
3. Generates $N$ (16–128) parametric boundary annotation points.
4. Performs temporal tracking and Exponential Moving Average (EMA) smoothing across video frames.
5. Measures Pupillary Light Reflex (PLR) Dilation Velocity ($\frac{d r_p}{d t}$) and constriction dynamics.
6. Exports Machine Learning training datasets in COCO Keypoints JSON and YOLO-Pose Keypoint TXT format.
7. Operates 100% locally on CPU without requiring paid APIs or GPUs.

---

## ✨ System Features

- **Starburst Radial Gradient Ray-Tracing & RANSAC Pruning**: Casts 48 radial rays outwards from pupil center to locate maximum gradient edges and uses RANSAC to filter out eyelash/shadow outliers.
- **Sectorial Daugman Integrodifferential Iris Search**: Integrates radial intensity gradients across non-occluded lateral sectors ($\theta \in [-50^\circ, +50^\circ] \cup [130^\circ, 230^\circ]$) with multi-scale Gaussian smoothing ($\sigma = 1, 2, 3$).
- **2D Eye Gaze Vector & Direction Tracking**: Measures 2D Gaze Displacement, Gaze Angle ($\theta_{gaze}$), and Cardinal Direction ("Center", "Looking Left", "Looking Up-Right", etc.) with visual Gaze Arrow overlays.
- **Pupillary Light Reflex (PLR) & Dilation Dynamics**: Tracks pupil dilation/constriction velocity ($\frac{d r_p}{d t}$) across video frames.
- **Frame-by-Frame Video Processing**: Supports MP4, AVI, MOV, MKV files; Fast, Balanced, and High Accuracy detection modes; frame skipping (1, 2, 5, 10).
- **Interactive Frame Explorer**: Slider to inspect any video frame with side-by-side Original vs Annotated comparison and status badges (`DETECTED`, `PREDICTED`, `FAILED`).
- **Multi-Format Export Suite**: Export PNG images, MP4/AVI videos, CSV metric reports, COCO Keypoints JSON, and YOLO Keypoints TXT.

---

## 🛠️ Technology Stack

- **Python 3.10+**
- **Streamlit**: Web frontend and multi-tab dashboard layout.
- **OpenCV (`opencv-python`)**: Image processing, CLAHE, glint inpainting, contours, edge detection, video I/O.
- **NumPy**: Matrix operations, Starburst ray tracing, RANSAC, and Gaze Vector math.
- **Pillow**: Image conversion.
- **Matplotlib**: Time-series tracking charts, PLR velocity plots, and intensity profiles.
- **pandas**: Data logging and CSV export.
- **imageio & imageio-ffmpeg**: Cross-platform video reading and encoding.

---

## 📁 Project Structure

```
eye_annotation/
├── app.py                      # Streamlit Dashboard (6 Navigation Tabs)
├── requirements.txt            # Python Dependencies
├── README.md                   # System Documentation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Adaptive preprocessing & glint inpainting
│   ├── quality.py              # Brightness, contrast, sharpness, blur & noise checks
│   ├── pupil_detection.py      # Pupil detection with Starburst RANSAC integration
│   ├── starburst_refinement.py # Starburst Ray-Tracing & RANSAC boundary refiner
│   ├── iris_detection.py       # Sectorial Daugman radial gradient iris search
│   ├── gaze_tracking.py        # 2D Gaze Vector calculation & arrow overlay renderer
│   ├── candidate_scoring.py    # Multi-candidate quality scoring & ranking
│   ├── validation.py           # Physical & temporal sanity checks
│   ├── annotation.py           # Parametric boundary point generator (16-128 points) & Gaze overlay
│   ├── tracking.py             # Temporal state tracking & prediction failure recovery
│   ├── smoothing.py            # Exponential Moving Average coordinate smoothing
│   ├── video_processor.py      # Frame-by-frame video processing engine
│   ├── confidence.py           # Multi-factor confidence score calculator
│   ├── analytics.py            # PLR dilation velocity & biometric decentration
│   ├── visualization.py        # Analytics charts & preprocessing grids
│   └── export.py               # PNG, MP4, CSV, JSON, COCO/YOLO keypoint exporter
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Image conversion & helper utilities
│   ├── video_utils.py          # Video metadata & frame generator
│   └── synthetic_eye.py        # Synthetic eye image & video generator
├── sample_images/              # Pre-packaged test images
├── sample_videos/              # Pre-packaged test videos (01_eyes.mp4)
├── outputs/                    # Output storage
└── tests/                      # Automated unit test suite (9 test modules)
```

---

## 💻 Installation & Quick Start

### 1. Navigate to Project Directory
```bash
cd "c:\Users\vanig\GENERATIVE AI ASSIGNMENTS\EYE ANNOTATION"
```

### 2. Create Virtual Environment
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

### 4. Run Automated Unit Tests
```bash
python -m unittest discover -s tests
```

### 5. Launch Application
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.
