import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from utils.synthetic_eye import generate_synthetic_eye

def create_sample_dataset():
    target_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(target_dir, exist_ok=True)
    
    samples = [
        ("eye_sample_1.jpg", 42, 115, (0, 0), (120, 75, 50), True, 0.03, 1),
        ("eye_sample_2.jpg", 55, 135, (10, -5), (60, 100, 140), True, 0.04, 1),
        ("eye_sample_3.jpg", 32, 95, (-15, 10), (90, 120, 60), True, 0.05, 3),
    ]
    
    for filename, pr, ir, offset, color, glint, noise, blur in samples:
        img_rgb, _ = generate_synthetic_eye(
            width=640,
            height=480,
            pupil_r=pr,
            iris_r=ir,
            pupil_offset=offset,
            iris_color=color,
            glint=glint,
            noise_level=noise,
            blur_level=blur
        )
        pil_img = Image.fromarray(img_rgb)
        filepath = os.path.join(target_dir, filename)
        pil_img.save(filepath, quality=95)
        print(f"Generated sample image: {filepath}")

if __name__ == "__main__":
    create_sample_dataset()
