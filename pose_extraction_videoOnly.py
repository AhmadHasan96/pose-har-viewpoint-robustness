import cv2
import mediapipe as mp
import os

# === SETTINGS ===
#VIDEO_PATH = "New_videos/04.05.2026/120Hz.mp4"
VIDEO_PATH = "videos/lifting_front_normal.mp4"
SAVE_DIR = "New_videos_MVP/"
os.makedirs(SAVE_DIR, exist_ok=True)

# === INIT ===
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
# min_detection_confidence=.90,
#     min_tracking_confidence=0.90
     min_detection_confidence=0.5,
     min_tracking_confidence=0.5
)

# === VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_path = os.path.join(SAVE_DIR, f"{video_name}_skeleton._mdc_0p5_mtc_0p5.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    draw_frame = frame.copy()

    if result.pose_landmarks:
        mp_draw.draw_landmarks(draw_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(100, 255, 0), thickness=2, circle_radius=1),
                               mp_draw.DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=4))

    # mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    # mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
    #
    out.write(draw_frame)

cap.release()
out.release()
pose.close()

print(f"✅ Saved skeleton video → {out_path}")