import os
import glob

temp_dir = os.environ.get("TEMP", "")
files = glob.glob(os.path.join(temp_dir, "*.mp4"))
for f in files:
    size = os.path.getsize(f)
    if "output" not in f and size > 100000:
        print(f"Candidate input: {f} (size={size})")
