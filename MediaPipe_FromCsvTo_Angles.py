import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mediapipe_csv_path = "New_videos/July/8Angles_Camera at BirdView-level_world_landmarks.csv"
output_DIR = "New_videos/July/MediaPipe Angels"
os.makedirs(output_DIR, exist_ok=True)

file_name = os.path.splitext(os.path.basename(mediapipe_csv_path))[0]
mpdf = pd.read_csv(mediapipe_csv_path)

print(f"Loaded: {mediapipe_csv_path}")
print(mpdf.shape)


# ==========================================
# 2) HELPER: ANGLE AT POINT B FROM A-B-C
# ==========================================
def angle_abc(A, B, C):
    BA = A - B
    BC = C - B

    dot = np.sum(BA * BC, axis=1)
    normBA = np.linalg.norm(BA, axis=1)
    normBC = np.linalg.norm(BC, axis=1)

    denom = normBA * normBC
    cosang = np.divide(dot, denom, out=np.full_like(dot, np.nan, dtype=float), where=denom != 0)
    cosang = np.clip(cosang, -1.0, 1.0)

    ang = np.degrees(np.arccos(cosang))
    return ang


# =========================
# 3) BUILD JOINT ARRAYS
# MediaPipe Pose indices:
# 11 L shoulder, 12 R shoulder
# 13 L elbow,    14 R elbow
# 23 L hip,      24 R hip
# 25 L knee,     26 R knee
# =========================
# Lsh = mpdf[['x_11', 'y_11', 'z_11']].to_numpy()
# Rsh = mpdf[['x_12', 'y_12', 'z_12']].to_numpy()

# Lel = mpdf[['x_13', 'y_13', 'z_13']].to_numpy()
# Rel = mpdf[['x_14', 'y_14', 'z_14']].to_numpy()

# Lhip = mpdf[['x_23', 'y_23', 'z_23']].to_numpy()
# Rhip = mpdf[['x_24', 'y_24', 'z_24']].to_numpy()

# Lknee = mpdf[['x_25', 'y_25', 'z_25']].to_numpy()
# Rknee = mpdf[['x_26', 'y_26', 'z_26']].to_numpy()

Lsh = mpdf[['x_11', 'y_11']].to_numpy()
Rsh = mpdf[['x_12', 'y_12']].to_numpy()

Lel = mpdf[['x_13', 'y_13']].to_numpy()
Rel = mpdf[['x_14', 'y_14']].to_numpy()

Lhip = mpdf[['x_23', 'y_23']].to_numpy()
Rhip = mpdf[['x_24', 'y_24']].to_numpy()

Lknee = mpdf[['x_25', 'y_25']].to_numpy()
Rknee = mpdf[['x_26', 'y_26']].to_numpy()


# =========================
# 4) COMPUTE 4 ANGLES
# =========================
L_shoulder_deg = angle_abc(Lhip, Lsh, Lel)
R_shoulder_deg = angle_abc(Rhip, Rsh, Rel)

L_hip_deg = angle_abc(Lsh, Lhip, Lknee)
R_hip_deg = angle_abc(Rsh, Rhip, Rknee)


# =========================
# 5) SAVE OUTPUT CSV
# =========================
time_col = 'rel_time_sec' if 'rel_time_sec' in mpdf.columns else 'time_s'

angles_df = pd.DataFrame({
    time_col: mpdf[time_col],
    'L_shoulder_deg': L_shoulder_deg,
    'R_shoulder_deg': R_shoulder_deg,
    'L_hip_deg': L_hip_deg,
    'R_hip_deg': R_hip_deg,
})

angles_csv_path = os.path.join(output_DIR, f"{file_name}_MP_4angles_2D.csv")
angles_df.to_csv(angles_csv_path, index=False)

print(f"Saved angles CSV: {angles_csv_path}")
angles_df.head()

