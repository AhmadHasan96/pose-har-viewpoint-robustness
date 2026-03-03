import cv2
import mediapipe as mp
import pandas as pd
import os

# === SETTINGS ===
VIDEO_PATH = "videos/squatting/squatting_side_normal_4.mp4"
SAVE_DIR = "data/raw_landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

# === INIT MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === PROCESS VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
landmarks_data = []

frame_count = 0
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(total_frames)
while True:
    # print(frame_count)
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        frame_landmarks = []
        for lm in result.pose_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        landmarks_data.append(frame_landmarks)

    frame_count += 1

cap.release()
pose.close()

# === SAVE TO CSV ===
if landmarks_data:
    columns = []
    for i in range(33):  # 33 landmarks
        columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"])
    df = pd.DataFrame(landmarks_data, columns=columns)
    output_path = os.path.join(SAVE_DIR, f"{video_name}_landmarks.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved landmarks for {video_name} → {output_path}")
else:
    print("⚠️ No pose landmarks detected. Check video quality or visibility.")
