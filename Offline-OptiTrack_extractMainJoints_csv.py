import pandas as pd
import os

# === SETTINGS ===
INPUT_CSV = "New_videos/04.05.2026/Opti_cut.csv"          # <-- set your file here
OUTPUT_CSV = "New_videos/04.05.2026/optitrack_main_bones_xyz.csv"

HIP_BONE = "FullBody:Hip"

MAIN_BONES = [
    "FullBody:Hip",
    "FullBody:Ab",
    "FullBody:Chest",
    "FullBody:Neck",
    "FullBody:Head",
    "FullBody:LShoulder",
    "FullBody:LUArm",
    "FullBody:LFArm",
    "FullBody:LHand",
    "FullBody:RShoulder",
    "FullBody:RUArm",
    "FullBody:RFArm",
    "FullBody:RHand",
    "FullBody:LThigh",
    "FullBody:LShin",
    "FullBody:LFoot",
    "FullBody:RThigh",
    "FullBody:RShin",
    "FullBody:RFoot",
]

# === READ OPTITRACK CSV (bones) ===
# Assumes standard Motive bones CSV with 7 meta header rows
raw = pd.read_csv(INPUT_CSV, header=None)

META_ROWS = 7
header_rows = raw.iloc[:META_ROWS].fillna("")
data = raw.iloc[META_ROWS:].reset_index(drop=True)

# Build flat column names like: FullBody:Hip_Position_X
col_names = []
for c in range(raw.shape[1]):
    type_row   = str(header_rows.iloc[2, c]).strip() if c < header_rows.shape[1] else ""
    bone_row   = str(header_rows.iloc[3, c]).strip() if c < header_rows.shape[1] else ""
    trans_row  = str(header_rows.iloc[5, c]).strip() if c < header_rows.shape[1] else ""
    axis_row   = str(header_rows.iloc[6, c]).strip() if c < header_rows.shape[1] else ""

    if c == 0:
        col_names.append("Frame")
    elif c == 1:
        col_names.append("Time_Seconds")
    elif bone_row and trans_row and axis_row:
        # e.g. FullBody:Hip_Position_X
        col_names.append(f"{bone_row}_{trans_row}_{axis_row}")
    else:
        col_names.append(f"col_{c}")

data.columns = col_names
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

hip_x = data[f"{HIP_BONE}_Position_X"]
hip_y = data[f"{HIP_BONE}_Position_Y"]
hip_z = data[f"{HIP_BONE}_Position_Z"]

out = pd.DataFrame()
out["Time_Seconds"] = data["Time_Seconds"]

for bone in MAIN_BONES:
        
    out[f"{bone}_X"] = data[f"{bone}_Position_X"]
    out[f"{bone}_Y"] = data[f"{bone}_Position_Y"]
    out[f"{bone}_Z"] = -data[f"{bone}_Position_Z"]

# === SAVE ===
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved main-bone positions → {OUTPUT_CSV}")


