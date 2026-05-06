import cv2
import mediapipe as mp
import pandas as pd
import os

# === SETTINGS ===
VIDEO_PATH = "New_videos/04.05.2026/120Hz.mp4"
SAVE_DIR = "New_videos/04.05.2026/raw_landmarks"
SAVE_VIDEO = True
os.makedirs(SAVE_DIR, exist_ok=True)

# === INIT MEDIAPIPE ===
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
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
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if SAVE_VIDEO:
    out_path = os.path.join(SAVE_DIR, f"{video_name}_skeleton.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    frame_landmarks = []

    if result.pose_world_landmarks:
        for lm in result.pose_world_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        for i in range(33):
            frame_landmarks.extend([None, None, None, None])

    landmarks_data.append(frame_landmarks)

    if SAVE_VIDEO:
        draw_frame = frame.copy()
        if result.pose_landmarks:
            mp_draw.draw_landmarks(draw_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        out.write(draw_frame)

    frame_count += 1

cap.release()
pose.close()

if SAVE_VIDEO:
    out.release()

# === SAVE TO CSV ===
if landmarks_data:
    columns = []
    for i in range(33):
        columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"])

    df = pd.DataFrame(landmarks_data, columns=columns)
    output_path = os.path.join(SAVE_DIR, f"{video_name}_world_landmarks.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved world landmarks for {video_name} → {output_path}")

    if SAVE_VIDEO:
        print(f"✅ Saved skeleton video → {out_path}")
else:
    print("⚠️ No pose detected.")