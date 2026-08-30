import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import time


start_time = time.perf_counter()

# === SETTINGS ===
VIDEO_PATH = "New_videos/August/DJI.mp4"
OptiTrack_CSV = "New_videos/August/Opti.csv"
SAVE_DIR = "New_videos/August/Output"
model_complexity = 1
os.makedirs(SAVE_DIR, exist_ok=True)


# === INIT MEDIAPIPE ===
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    model_complexity = model_complexity,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,

)

# === PROCESS VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
landmarks_data = []

fps = 60.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video_path = os.path.join(SAVE_DIR, f"{video_name}_MediaPipe_skeleton_{model_complexity}.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    frame_time_s = frame_count / fps
    frame_landmarks = [frame_count, frame_time_s]

    annotated = frame.copy()

    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            annotated,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

    if result.pose_world_landmarks:
        for lm in result.pose_world_landmarks.landmark:
            frame_landmarks.extend([lm.x, -lm.y, lm.z, lm.visibility])
    else:
        for i in range(33):
            frame_landmarks.extend([None, None, None, None])

    writer.write(annotated)
    landmarks_data.append(frame_landmarks)
    frame_count += 1

cap.release()
writer.release()
pose.close()

# === SAVE FULL RAW CSV ===
if not landmarks_data:
    print("⚠️ No pose landmarks detected.")
    raise SystemExit

columns = ["frame", "time_s"]
for i in range(33):
    columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"])

df = pd.DataFrame(landmarks_data, columns=columns)

output_csv_path = os.path.join(SAVE_DIR, f"{video_name}_MediaPipe_rawData_{model_complexity}.csv")
df.to_csv(output_csv_path, index=False)

print(f"✅ Saved full raw CSV → {output_csv_path}")
print(f"✅ Saved skeleton video → {out_video_path}")


################################################# Calculating Angles ########################################


# =========================
# 1) LOAD SAVED MP RAW CSV
# =========================
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
# raw_csv_path = os.path.join(SEGMENTS_DIR, seg_name) ### Uncomment me when you uncomment segmentation
Angles_DIR = os.path.join(SAVE_DIR, f"MediaPipe Angles Calculation_{model_complexity}")
os.makedirs(Angles_DIR, exist_ok=True)

# mpdf = pd.read_csv(raw_csv_path) ### Uncomment me when you uncomment segmentation
mpdf = pd.read_csv(output_csv_path) ### comment me when you uncomment segmentation

# print(f"Loaded: {raw_csv_path}")  ### Uncomment me when you uncomment segmentation
print(f"Loaded: {output_csv_path}") ### comment me when you uncomment segmentation
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
Lsh = mpdf[['x_11', 'y_11', 'z_11']].to_numpy()
Rsh = mpdf[['x_12', 'y_12', 'z_12']].to_numpy()

Lel = mpdf[['x_13', 'y_13', 'z_13']].to_numpy()
Rel = mpdf[['x_14', 'y_14', 'z_14']].to_numpy()

Lhip = mpdf[['x_23', 'y_23', 'z_23']].to_numpy()
Rhip = mpdf[['x_24', 'y_24', 'z_24']].to_numpy()

Lknee = mpdf[['x_25', 'y_25', 'z_25']].to_numpy()
Rknee = mpdf[['x_26', 'y_26', 'z_26']].to_numpy()


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

angles_csv_path = os.path.join(Angles_DIR, f"{video_name}_MP_4angles_3D.csv")
angles_df.to_csv(angles_csv_path, index=False)

print(f"Saved angles CSV: {angles_csv_path}")
angles_df.head()

end_time = time.perf_counter()
elapsed_seconds = end_time - start_time

print(f"Processing time: {elapsed_seconds:.2f} seconds")

Angles_DIR = os.path.join(SAVE_DIR, f"MediaPipe Angles Calculation_{model_complexity}")
angles_csv_path = os.path.join(Angles_DIR, "90_deg_MP_4angles_3D.csv")

############################################   OpTiTrack      #################################################
########################################### Hip Centering + Segmentation   # ##############################################
# import csv
# import copy
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# import pandas as pd

# # # ========= SETTINGS =========

# #  = "New_videos/June/12.06.2026/one_angle_take_hip_centered_raw.csv"
# csv_name = os.path.splitext(os.path.basename(OptiTrack_CSV))[0]
# OUTPUT_CSV = os.path.join(SAVE_DIR, f"{csv_name}_hip_centered_rawData.csv")

# NAME_OF_SKELETON = "Ahmed_full_body" #"Skeleton"   # or "FullBody"
# HEADER_ROWS_COUNT = 7

# KEEP_HIP_AS_ZERO = True
# REMOVE_BONE_MARKERS = True

# SMOOTH_WINDOW = 5
# MIN_TIME_BETWEEN_PEAKS = 0.8
# MIN_PEAK_HEIGHT = 0.8
# MIN_PEAK_PROMINENCE = 0.05

# SKIP_AFTER_JUMP = 2.0
# STOP_BEFORE_NEXT_JUMP = 3.0
# # ============================

# os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
# os.makedirs(SAVE_DIR, exist_ok=True)

# # === READ RAW CSV ===
# rows = []
# with open(OptiTrack_CSV, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         rows.append(row)

# max_cols = max(len(r) for r in rows)
# rows = [r + [""] * (max_cols - len(r)) for r in rows]

# header_rows = rows[:HEADER_ROWS_COUNT]
# data_rows = rows[HEADER_ROWS_COUNT:]

# type_row = header_rows[2]
# name_row = header_rows[3]
# id_row = header_rows[4]
# transform_row = header_rows[5]
# axis_row = header_rows[6]

# # === FIND HIP POSITION COLUMNS ===
# hip_cols = {"X": None, "Y": None, "Z": None}

# for c in range(max_cols):
#     name_val = name_row[c].strip()
#     transform_val = transform_row[c].strip()
#     axis_val = axis_row[c].strip()

#     if name_val == f"{NAME_OF_SKELETON}:Hip" and transform_val == "Position" and axis_val in hip_cols:
#         hip_cols[axis_val] = c

# missing_hip = [ax for ax, col in hip_cols.items() if col is None]
# if missing_hip:
#     raise ValueError(f"Missing hip position columns for axes: {missing_hip}")

# print("Hip columns:", hip_cols)

# # === FIND ALL POSITION COLUMNS TO CENTER ===
# position_cols = {"X": [], "Y": [], "Z": []}

# for c in range(max_cols):
#     name_val = name_row[c].strip()
#     transform_val = transform_row[c].strip()
#     axis_val = axis_row[c].strip()
#     type_val = type_row[c].strip()

#     if transform_val == "Position" and axis_val in position_cols:
#         if REMOVE_BONE_MARKERS:
#             if type_val == "Bone" and name_val.startswith(f"{NAME_OF_SKELETON}:"):
#                 position_cols[axis_val].append(c)
#         else:
#             position_cols[axis_val].append(c)

# print("Position columns found:", {k: len(v) for k, v in position_cols.items()})

# # === OPTIONAL COLUMN FILTERING ===
# keep_cols = list(range(max_cols))

# if REMOVE_BONE_MARKERS:
#     filtered_cols = []
#     for c in range(max_cols):
#         type_val = type_row[c].strip()
#         name_val = name_row[c].strip()
#         transform_val = transform_row[c].strip()
#         axis_val = axis_row[c].strip()

#         keep = True

#         if transform_val == "Position" and axis_val in {"X", "Y", "Z"}:
#             if not (type_val == "Bone" and name_val.startswith(f"{NAME_OF_SKELETON}:")):
#                 keep = False

#         if keep:
#             filtered_cols.append(c)

#     keep_cols = filtered_cols

# # === APPLY HIP CENTERING ===
# new_data_rows = []

# for row in data_rows:
#     new_row = copy.copy(row)

#     try:
#         hip_x = float(row[hip_cols["X"]]) if row[hip_cols["X"]] != "" else None
#         hip_y = float(row[hip_cols["Y"]]) if row[hip_cols["Y"]] != "" else None
#         hip_z = float(row[hip_cols["Z"]]) if row[hip_cols["Z"]] != "" else None
#     except ValueError:
#         hip_x, hip_y, hip_z = None, None, None

#     if hip_x is not None and hip_y is not None and hip_z is not None:
#         for c in position_cols["X"]:
#             try:
#                 if new_row[c] != "":
#                     new_row[c] = str(float(new_row[c]) - hip_x)
#             except ValueError:
#                 pass

#         for c in position_cols["Y"]:
#             try:
#                 if new_row[c] != "":
#                     new_row[c] = str(float(new_row[c]) - hip_y)
#             except ValueError:
#                 pass

#         for c in position_cols["Z"]:
#             try:
#                 if new_row[c] != "":
#                     new_row[c] = str(float(new_row[c]) - hip_z)
#             except ValueError:
#                 pass

#         if KEEP_HIP_AS_ZERO:
#             new_row[hip_cols["X"]] = "0.0"
#             new_row[hip_cols["Y"]] = "0.0"
#             new_row[hip_cols["Z"]] = "0.0"

#     new_data_rows.append(new_row)

# # === FILTER COLUMNS IF REQUESTED ===
# final_rows = []
# for row in header_rows + new_data_rows:
#     final_rows.append([row[c] for c in keep_cols])

# # === WRITE CENTERED CSV ===
# with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(final_rows)

# print(f"Saved hip-centered raw CSV to: {OUTPUT_CSV}")


# ############################################ Angles Calculation #########################################

# # =========================
# # SETTINGS
# # =========================
# print(OUTPUT_CSV)
# INPUT_SEGMENT_CSV = OUTPUT_CSV

# Opti_Angles_DIR = os.path.join(SAVE_DIR, "OptiTrack Angles Calculation")
# os.makedirs(Opti_Angles_DIR, exist_ok=True)

# NAME_OF_SKELETON = "Ahmed_full_body"#"Skeleton"   # or "FullBody"
# HEADER_ROWS_COUNT = 7
# # =========================


# def angle_3d(a, b, c):
#     a = np.asarray(a, dtype=float)
#     b = np.asarray(b, dtype=float)
#     c = np.asarray(c, dtype=float)

#     ba = a - b
#     bc = c - b

#     ba_norm = np.linalg.norm(ba, axis=1)
#     bc_norm = np.linalg.norm(bc, axis=1)

#     dot = np.sum(ba * bc, axis=1)
#     cosang = dot / np.clip(ba_norm * bc_norm, 1e-12, None)
#     cosang = np.clip(cosang, -1.0, 1.0)

#     return np.degrees(np.arccos(cosang))


# def find_col_indices(header_rows, bone_name):
#     name_row = header_rows[3]
#     transform_row = header_rows[5]
#     axis_row = header_rows[6]

#     cols = {"X": None, "Y": None, "Z": None}

#     for i in range(len(name_row)):
#         name_val = name_row[i].strip()
#         transform_val = transform_row[i].strip()
#         axis_val = axis_row[i].strip()

#         if name_val == f"{NAME_OF_SKELETON}:{bone_name}" and transform_val == "Position" and axis_val in cols:
#             cols[axis_val] = i

#     missing = [k for k, v in cols.items() if v is None]
#     if missing:
#         raise ValueError(f"Missing {bone_name} position columns: {missing}")

#     return cols


# def get_xyz_array(df, cols):
#     return df.iloc[:, [cols["X"], cols["Y"], cols["Z"]]].apply(pd.to_numeric, errors="coerce").to_numpy()


# # --- read raw segment csv with 7 header rows
# rows = []
# with open(INPUT_SEGMENT_CSV, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         rows.append(row)

# max_cols = max(len(r) for r in rows)
# rows = [r + [""] * (max_cols - len(r)) for r in rows]

# header_rows = rows[:HEADER_ROWS_COUNT]
# data_rows = rows[HEADER_ROWS_COUNT:]

# # dataframe with original column order preserved
# df = pd.DataFrame(data_rows)

# # time column
# time_col = None
# axis_row = header_rows[6]
# for i in range(len(axis_row)):
#     if axis_row[i].strip() == "Time (Seconds)":
#         time_col = i
#         break

# if time_col is None:
#     raise ValueError("Could not find Time (Seconds) column.")

# time_vals = pd.to_numeric(df.iloc[:, time_col], errors="coerce").to_numpy()



# # --- your 8 replacement points from bone centers

# luarm_cols  = find_col_indices(header_rows, "LUArm")
# print(luarm_cols)
# ruarm_cols  = find_col_indices(header_rows, "RUArm")

# lfarm_cols  = find_col_indices(header_rows, "LFArm")
# rfarm_cols  = find_col_indices(header_rows, "RFArm")

# lthigh_cols = find_col_indices(header_rows, "LThigh")
# rthigh_cols = find_col_indices(header_rows, "RThigh")

# lshin_cols  = find_col_indices(header_rows, "LShin")
# rshin_cols  = find_col_indices(header_rows, "RShin")

# # positions

# luarm  = get_xyz_array(df, luarm_cols)
# ruarm  = get_xyz_array(df, ruarm_cols)

# lfarm  = get_xyz_array(df, lfarm_cols)
# rfarm  = get_xyz_array(df, rfarm_cols)


# lthigh = get_xyz_array(df, lthigh_cols)
# rthigh = get_xyz_array(df, rthigh_cols)


# lshin  = get_xyz_array(df, lshin_cols)
# rshin  = get_xyz_array(df, rshin_cols)


# # --- angle calculation from 3 points
# # centered at the replacement point you specified
# L_shoulder_deg = angle_3d(lthigh,  luarm,  lfarm)
# R_shoulder_deg = angle_3d(rthigh,  ruarm,  rfarm)

# L_hip_deg      = angle_3d(luarm,    lthigh, lshin)
# R_hip_deg      = angle_3d(ruarm,    rthigh, rshin)


# # keep valid rows only
# out_df = pd.DataFrame({
#     "time_s": time_vals,
#     "L_shoulder_deg": L_shoulder_deg,
#     "R_shoulder_deg": R_shoulder_deg,
#     "L_hip_deg": L_hip_deg,
#     "R_hip_deg": R_hip_deg,

# })

# out_df = out_df.dropna().reset_index(drop=True)

# os.makedirs(os.path.dirname(Opti_Angles_DIR), exist_ok=True)
# out_csv = os.path.join(Opti_Angles_DIR, f"{csv_name}_4angles_3D.csv")
# out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# print(f"Saved angle CSV -> {Opti_Angles_DIR}")
# print(out_df.head())