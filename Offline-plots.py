import pandas as pd
import matplotlib.pyplot as plt

# ========= FILES =========
MP_CSV = "New_videos/04.05.2026/MediaPipe_main_joints_named.csv"
OPTI_CSV = "New_videos/04.05.2026/optitrack_main_bones_xyz.csv" 

mp = pd.read_csv(MP_CSV)
opti = pd.read_csv(OPTI_CSV)

# ========= SAME SEMANTIC ORDER =========
parts = [
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
]

# ========= SHARED COLORS =========
colors = {
    "head":           "#000000",  # black
    "left_shoulder":  "#e41a1c",  # red
    "left_elbow":     "#377eb8",  # blue
    "left_wrist":     "#4daf4a",  # green
    "left_hip":       "#984ea3",  # purple
    "left_knee":      "#ff7f00",  # orange
    "left_ankle":     "#ffff33",  # yellow
    "right_shoulder": "#a65628",  # brown
    "right_elbow":    "#f781bf",  # pink
    "right_wrist":    "#17becf",  # cyan
    "right_hip":      "#999999",  # gray
    "right_knee":     "#3700ff",  # 
    "right_ankle":    "#c3cb21",  #
}

# ========= COLUMN NAME MAPPING =========
# MP uses joints
mp_cols = {
    "head":           "nose",
    "left_shoulder":  "left_shoulder",
    "left_elbow":     "left_elbow",
    "left_wrist":     "left_wrist",
    "left_hip":       "left_hip",
    "left_knee":      "left_knee",
    "left_ankle":     "left_ankle",
    "right_shoulder": "right_shoulder",
    "right_elbow":    "right_elbow",
    "right_wrist":    "right_wrist",
    "right_hip":      "right_hip",
    "right_knee":     "right_knee",
    "right_ankle":    "right_ankle",
}

# Opti uses bones / your naming
opti_cols = {
    "head":           "FullBody:Head",
    "left_shoulder":  "FullBody:LUArm",
    "left_elbow":     "FullBody:LFArm",
    "left_wrist":     "FullBody:LHand",
    "left_hip":       "FullBody:LThigh",
    "left_knee":      "FullBody:LShin",
    "left_ankle":     "FullBody:LFoot",
    "right_shoulder": "FullBody:RUArm",
    "right_elbow":    "FullBody:RFArm",
    "right_wrist":    "FullBody:RHand",
    "right_hip":      "FullBody:RThigh",
    "right_knee":     "FullBody:RShin",
    "right_ankle":    "FullBody:RFoot",
}

def plot_dataset(df, time_col, colmap, title_prefix):
    for axis in ["x", "y", "z"]:
        plt.figure(figsize=(12, 4))
        for part in parts:
            if title_prefix == "MediaPipe":
                col = f"{colmap[part]}_{axis}"
            else:
                col = f"{colmap[part]}_{axis.upper()}"

            if col in df.columns:
                plt.plot(df[time_col], df[col], label=part, color=colors[part])

        plt.xlabel(time_col)
        plt.ylabel(f"{axis.upper()} position")
        plt.title(f"{title_prefix} – {axis.upper()} axis")
        plt.legend(fontsize=8, ncol=3)
        plt.grid(True)
        plt.tight_layout()

# ========= MAKE 6 PLOTS =========
plot_dataset(mp, "time_ms", mp_cols, "MediaPipe")
plot_dataset(opti, "Time_Seconds", opti_cols, "OptiTrack")

plt.show()