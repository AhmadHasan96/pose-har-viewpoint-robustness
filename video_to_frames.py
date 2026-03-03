# video_to_frames.py
import cv2
import os

# ===== CONFIG =====
video_path = "skeleton_videos/skeleton_lifting_side_normal_2.mp4"  # path to your video
output_dir = "frames/lifting_side_normal_2"               # folder to save frames

os.makedirs(output_dir, exist_ok=True)
# ==================

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Could not open video: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"🎥 Extracting {frame_count} frames at {fps} FPS...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create filename like frame_0001.jpg
    frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    count += 1
    if count % 100 == 0:
        print(f"✅ Saved {count} frames...")

cap.release()
print(f"✅ Done! Total frames saved: {count}")
print(f"📁 Frames stored in: {os.path.abspath(output_dir)}")
