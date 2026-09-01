import matplotlib.pyplot as plt
import io
import numpy as np
import cv2
from typing import Dict, Any
from PIL import Image

def generate_intensity_profile_plot(
    profile_x: list,
    profile_intensity: list,
    pupil_x: int,
    pupil_r: float,
    iris_x: int,
    iris_r: float
) -> bytes:
    """Renders Matplotlib figure of horizontal intensity profile across eye pupil & iris."""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')

    ax.plot(profile_x, profile_intensity, color='#1f77b4', linewidth=1.8, label='Intensity (0-255)')

    # Highlight pupil region
    p_min = max(0, int(pupil_x - pupil_r))
    p_max = int(pupil_x + pupil_r)
    ax.axvspan(p_min, p_max, color='#17becf', alpha=0.25, label='Pupil Region')

    # Highlight iris region
    i_min = max(0, int(iris_x - iris_r))
    i_max = int(iris_x + iris_r)
    ax.axvspan(i_min, p_min, color='#e377c2', alpha=0.15, label='Iris Region')
    ax.axvspan(p_max, i_max, color='#e377c2', alpha=0.15)

    ax.axvline(x=pupil_x, color='#d62728', linestyle='--', label='Pupil Center')

    ax.set_title("Horizontal Intensity Profile Across Eye Center", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Pixel X Coordinate", fontsize=9)
    ax.set_ylabel("Grayscale Intensity", fontsize=9)
    ax.set_ylim(0, 260)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8, frameon=True)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def create_preprocessing_thumbnail_grid(prep_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Prepares RGB representations of preprocessing pipeline stages for display."""
    return {
        "1. Original RGB": prep_dict["original_rgb"],
        "2. Grayscale": cv2.cvtColor(prep_dict["gray"], cv2.COLOR_GRAY2RGB),
        "3. Glint Mask": cv2.cvtColor(prep_dict["glint_mask"], cv2.COLOR_GRAY2RGB),
        "4. Inpainted & CLAHE": cv2.cvtColor(prep_dict["clahe"], cv2.COLOR_GRAY2RGB),
        "5. Bilateral Filter": cv2.cvtColor(prep_dict["bilateral"], cv2.COLOR_GRAY2RGB)
    }
