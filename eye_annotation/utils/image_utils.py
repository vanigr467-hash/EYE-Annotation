import io
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from typing import Tuple, Dict, Any, List

def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Reads image bytes and returns RGB NumPy array."""
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(pil_img)

def numpy_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    """Converts RGB NumPy array to PNG bytes for downloading or Streamlit display."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def resize_if_large(img_rgb: np.ndarray, max_dim: int = 1000) -> Tuple[np.ndarray, float]:
    """Resizes image preserving aspect ratio if dimensions exceed max_dim."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return img_rgb, 1.0
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def export_results_to_csv(detection_data: Dict[str, Any]) -> str:
    """Converts detection metrics into CSV string."""
    df = pd.DataFrame([detection_data])
    return df.to_csv(index=False)

def export_points_to_json(points_dict: Dict[str, Any]) -> str:
    """Converts detected boundary coordinates into JSON string."""
    return json.dumps(points_dict, indent=2)

def create_batch_zip(results_list: List[Dict[str, Any]]) -> bytes:
    """
    Creates a ZIP archive containing all annotated images (PNG) and summary CSV report.
    Each element of results_list is a dict with 'filename', 'image_rgb', 'metrics'.
    """
    zip_buf = io.BytesIO()
    all_metrics = []
    
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for res in results_list:
            fname = res.get("filename", "eye_annotated.png")
            img_bytes = numpy_to_png_bytes(res["image_rgb"])
            zip_file.writestr(f"annotated/{fname}", img_bytes)
            
            m = res["metrics"].copy()
            m["filename"] = fname
            all_metrics.append(m)
            
        if all_metrics:
            df_summary = pd.DataFrame(all_metrics)
            csv_str = df_summary.to_csv(index=False)
            zip_file.writestr("batch_summary_results.csv", csv_str)
            
    return zip_buf.getvalue()
