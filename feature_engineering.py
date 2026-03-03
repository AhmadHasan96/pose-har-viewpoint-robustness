import pandas as pd
import numpy as np
import os
import math

# === SETTINGS ===
CSV_PATH = "data/raw_landmarks/squatting_side_normal_4_landmarks.csv"
SAVE_DIR = "data/features"
os.makedirs(SAVE_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
video_name = os.path.splitext(os.path.basename(CSV_PATH))[0]

# === HELPER FUNCTIONS ===
def angle_3pts(a, b, c):
    """Compute the angle (in degrees) formed at point b by a-b-c."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

# === COMPUTE FEATURES PER FRAME ===
features = []

for i in range(len(df)):
    row = df.iloc[i]
    # convert to (x,y,z) numpy arrays
    def p(idx): return np.array([row[f"x_{idx}"], row[f"y_{idx}"], row[f"z_{idx}"]])

    # Key joints
    L_shoulder, R_shoulder = p(11), p(12)
    L_elbow, R_elbow = p(13), p(14)
    L_wrist, R_wrist = p(15), p(16)
    L_hip, R_hip = p(23), p(24)
    L_knee, R_knee = p(25), p(26)
    L_ankle, R_ankle = p(27), p(28)

    # Angles
    left_elbow_angle = angle_3pts(L_shoulder, L_elbow, L_wrist)
    right_elbow_angle = angle_3pts(R_shoulder, R_elbow, R_wrist)
    left_knee_angle = angle_3pts(L_hip, L_knee, L_ankle)
    right_knee_angle = angle_3pts(R_hip, R_knee, R_ankle)

    # Torso extension (distance shoulder ↔ hip)
    torso_length = (euclidean(L_shoulder, L_hip) + euclidean(R_shoulder, R_hip)) / 2

    # Speeds (if not first frame)
    if i > 0:
        prev = df.iloc[i-1]
        def prev_p(idx): return np.array([prev[f"x_{idx}"], prev[f"y_{idx}"], prev[f"z_{idx}"]])
        left_wrist_speed = euclidean(p(15), prev_p(15))
        right_wrist_speed = euclidean(p(16), prev_p(16))
    else:
        left_wrist_speed = right_wrist_speed = 0.0

    features.append([
        left_elbow_angle, right_elbow_angle,
        left_knee_angle, right_knee_angle,
        torso_length, left_wrist_speed, right_wrist_speed
    ])

# === SAVE FEATURES ===
cols = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "torso_length", "left_wrist_speed", "right_wrist_speed"
]
features_df = pd.DataFrame(features, columns=cols)
out_path = os.path.join(SAVE_DIR, f"{video_name}_features.csv")
features_df.to_csv(out_path, index=False)
print(f"✅ Features saved: {out_path}")
