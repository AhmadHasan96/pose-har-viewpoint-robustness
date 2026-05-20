import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= FILES =========
MP_CSV = "New_videos/04.05.2026/Human Movement seperated/BendingForward/MediaPipe_BendingForward.csv"
OPTI_CSV = "New_videos/04.05.2026/Human Movement seperated/BendingForward/OptiTrack_BendingForward.csv" 

# MP_CSV = "New_videos/04.05.2026/MediaPipe_main_joints_named.csv"
# OPTI_CSV = "New_videos/04.05.2026/optitrack_NoOffset_main_bones_xyz.csv" 

mp = pd.read_csv(MP_CSV)
opti = pd.read_csv(OPTI_CSV)

# ========= MATCHED JOINTS =========
# semantic name : (mp base name, opti base name)
joint_map = {
    "left_shoulder":  ("left_shoulder",  "FullBody:LUArm"),
    "left_elbow":     ("left_elbow",     "FullBody:LFArm"),
    "left_wrist":     ("left_wrist",     "FullBody:LHand"),
    "left_hip":       ("left_hip",       "FullBody:LThigh"),
    "left_knee":      ("left_knee",      "FullBody:LShin"),
    "left_ankle":     ("left_ankle",     "FullBody:LFoot"),
    "right_shoulder": ("right_shoulder", "FullBody:RUArm"),
    "right_elbow":    ("right_elbow",    "FullBody:RFArm"),
    "right_wrist":    ("right_wrist",    "FullBody:RHand"),
    "right_hip":      ("right_hip",      "FullBody:RThigh"),
    "right_knee":     ("right_knee",     "FullBody:RShin"),
    "right_ankle":    ("right_ankle",    "FullBody:RFoot"),
}

# ========= OPTIONAL: SAME LENGTH =========
n = min(len(mp), len(opti))
mp = mp.iloc[:n].reset_index(drop=True)
opti = opti.iloc[:n].reset_index(drop=True)

# ========= SELECT JOINTS TO PLOT =========
# selected_joints = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
# selected_joints = ["left_shoulder"]
selected_joints = ["left_knee"]

# ========= PLOT =========
fig, axs = plt.subplots(len(selected_joints), 3, figsize=(10, 3.5 * len(selected_joints)), sharex=True)

if len(selected_joints) == 1:
    axs = np.array([axs])

for i, joint_name in enumerate(selected_joints):
    mp_name, opti_name = joint_map[joint_name]

    mp_xyz = mp[[f"{mp_name}_x", f"{mp_name}_y", f"{mp_name}_z"]].to_numpy(dtype=float)
    opti_xyz = opti[[f"{opti_name}_X", f"{opti_name}_Y", f"{opti_name}_Z"]].to_numpy(dtype=float)

    diff = mp_xyz - opti_xyz

    # mask = ~np.isnan(diff).any(axis=1)
    # diff = diff[mask]

    # mp_xyz = mp_xyz[mask]
    # opti_xyz = opti_xyz[mask]
    # if diff.size == 0:
    #     print(joint_name, "has no valid frames, skipping")
    #     continue

    # time = np.arange(len(diff))
    time = mp["time_ms"]

    axis_labels = ["X", "Y", "Z"]

    for j in range(3):
        ax = axs[i, j]

        raw_diff = diff[:, j]
        mp_new = mp_xyz[:, j]
        opti_new = opti_xyz[:, j]
        # abs_err = np.abs(raw_diff)

        ax.plot(time, raw_diff, label="diff (MP - Opti)", color="blue", alpha=0.8)
        ax.plot(time, mp_new, label="MP",  alpha=0.8)
        ax.plot(time, opti_new, label="Opti", alpha=0.8)
        # ax.plot(time, abs_err, label="|diff|", color="red", alpha=0.6)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{joint_name} - {axis_labels[j]}")
        ax.set_ylabel("error [m]")
        ax.grid(True, alpha=0.3)

        if i == len(selected_joints) - 1:
            ax.set_xlabel("Time [s]")

        if i == 0 and j == 0:
            ax.legend(
            fontsize=8,
            # loc="lower left",
            # loc= "upper right",
            loc= "center right",
            frameon=True,
            borderpad=0.2,    # padding inside box
            labelspacing=0.2, # vertical space between labels
            handlelength=1.0, # length of line symbol
)

plt.tight_layout()
plt.show()