import cv2
import mediapipe as mp
import pandas as pd
import os

# === SETTINGS ===
VIDEO_PATH = "New_videos/04.05.2026/60Hz.mp4"
SAVE_DIR = "New_videos/04.05.2026/raw_landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

# === INIT MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === PROCESS VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
landmarks_data = []

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # frame_time_ms = (frame_count / fps) * 1000.0
    frame_time_s = (frame_count / 60)
    frame_landmarks = [frame_count, frame_time_s]

    if result.pose_world_landmarks:
        for lm in result.pose_world_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        for i in range(33):
            frame_landmarks.extend([None, None, None, None])

    landmarks_data.append(frame_landmarks)
    frame_count += 1

cap.release()
pose.close()

# === SAVE TO CSV ===
if landmarks_data:
    columns = ["frame", "time_ms"]
    for i in range(33):
        columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"])

    df = pd.DataFrame(landmarks_data, columns=columns)
    output_path = os.path.join(SAVE_DIR, f"{video_name}_world_landmarks.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved CSV → {output_path}")
else:
    print("⚠️ No pose landmarks detected.")