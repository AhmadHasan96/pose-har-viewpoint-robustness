import cv2
import os
from datetime import datetime

# === SETTINGS ===
SAVE_DIR = "videos"
FPS = 30
RESOLUTION = (640, 480)   # width x height
CLIP_DURATION = 15        # seconds

# === SETUP ===
movement = input("Enter movement name (e.g., lifting, twisting): ").strip().lower()
angle = input("Enter angle (front/side): ").strip().lower()
light = input("Enter lighting (normal/dim): ").strip().lower()

folder_path = os.path.join(SAVE_DIR, movement)
os.makedirs(folder_path, exist_ok=True)

# Count existing clips to name the next one correctly
existing = len([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
filename = f"{movement}_{angle}_{light}_{existing+1}.mp4"
output_path = os.path.join(folder_path, filename)

# === CAPTURE ===
cap = cv2.VideoCapture(0)
cap.set(3, RESOLUTION[0])
cap.set(4, RESOLUTION[1])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, FPS, RESOLUTION)

print(f"\n🎬 Recording '{filename}' ... Press 'q' to stop early.\n")
frame_count = 0
max_frames = CLIP_DURATION * FPS

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected!")
        break

    out.write(frame)
    cv2.imshow("Recording", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= max_frames:
        break

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Saved: {output_path}")
