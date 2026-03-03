# pose_skeleton_on_video.py
import cv2
import mediapipe as mp
import os

# ====== CONFIG ======
input_path = "videos/lifting/lifting_side_normal_2.mp4"  # path to your input video
SAVE_DIR = "skeleton_videos"                              # folder to save output
os.makedirs(SAVE_DIR, exist_ok=True)

# Create output file path
video_name = os.path.basename(input_path)
output_path = os.path.join(SAVE_DIR, f"skeleton_{video_name}")
# =====================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Could not open video: {input_path}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize MediaPipe Pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        # Write to output
        out.write(frame)

        # Optional live preview
        # cv2.imshow("Pose Skeleton Overlay", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Skeleton overlay video saved at: {os.path.abspath(output_path)}")
