import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.synthetic_eye import generate_synthetic_eye_video

def main():
    target_dir = os.path.dirname(__file__)
    output_video_path = os.path.join(target_dir, "01_eyes.mp4")
    print(f"Generating sample test video: {output_video_path}")
    generate_synthetic_eye_video(output_video_path, total_frames=90, fps=25.0)
    print(f"Successfully generated: {output_video_path}")

if __name__ == "__main__":
    main()
