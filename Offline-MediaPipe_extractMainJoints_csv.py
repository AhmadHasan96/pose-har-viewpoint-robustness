import pandas as pd
import os

# === SETTINGS ===
INPUT_CSV = "New_videos/04.05.2026/MP_cut.csv"
OUTPUT_CSV = "New_videos/04.05.2026/MediaPipe_main_joints_named.csv"

# === LOAD ===
df = pd.read_csv(INPUT_CSV)

# === KEEP ONLY MAIN JOINTS WITH NAMED COLUMNS ===
out = pd.DataFrame()

# keep time column
out["time_ms"] = df["time_ms"]

out["nose_x"] = df["x_0"]
out["nose_y"] = -df["y_0"]
out["nose_z"] = df["z_0"]
# left side
out["left_shoulder_x"] = df["x_11"]
out["left_shoulder_y"] = -df["y_11"]
out["left_shoulder_z"] = df["z_11"]

out["left_elbow_x"] = df["x_13"]
out["left_elbow_y"] = -df["y_13"]
out["left_elbow_z"] = df["z_13"]

out["left_wrist_x"] = df["x_15"]
out["left_wrist_y"] = -df["y_15"]
out["left_wrist_z"] = df["z_15"]

out["left_hip_x"] = df["x_23"]
out["left_hip_y"] = -df["y_23"]
out["left_hip_z"] = df["z_23"]

out["left_knee_x"] = df["x_25"]
out["left_knee_y"] = -df["y_25"]
out["left_knee_z"] = df["z_25"]

out["left_ankle_x"] = df["x_27"]
out["left_ankle_y"] = -df["y_27"]
out["left_ankle_z"] = df["z_27"]

# right side
out["right_shoulder_x"] = df["x_12"]
out["right_shoulder_y"] = -df["y_12"]
out["right_shoulder_z"] = df["z_12"]

out["right_elbow_x"] = df["x_14"]
out["right_elbow_y"] = -df["y_14"]
out["right_elbow_z"] = df["z_14"]

out["right_wrist_x"] = df["x_16"]
out["right_wrist_y"] = -df["y_16"]
out["right_wrist_z"] = df["z_16"]

out["right_hip_x"] = df["x_24"]
out["right_hip_y"] = -df["y_24"]
out["right_hip_z"] = df["z_24"]

out["right_knee_x"] = df["x_26"]
out["right_knee_y"] = -df["y_26"]
out["right_knee_z"] = df["z_26"]

out["right_ankle_x"] = df["x_28"]
out["right_ankle_y"] = -df["y_28"]
out["right_ankle_z"] = df["z_28"]

# === SAVE ===
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved named main joints CSV → {OUTPUT_CSV}")