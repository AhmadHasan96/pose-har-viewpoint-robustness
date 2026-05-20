import pandas as pd
import matplotlib.pyplot as plt

# === SETTINGS ===
INPUT_CSV = "New_videos/04.05.2026/optitrack_World_main_bones_xyz.csv"   # output from your previous transform

# File = "Local"
# File = "World"

IMPORTANT_BONES = [
    # "FullBody:Hip",
    # "FullBody:Ab",
    # "FullBody:Chest",
    # "FullBody:Neck",
    # "FullBody:Head",
    # "FullBody:LShoulder",
    "FullBody:LUArm",
    "FullBody:LFArm",
    "FullBody:LHand",
    # "FullBody:RShoulder",
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

# === LOAD ===
df = pd.read_csv(INPUT_CSV)

t = df["Time_Seconds"]

def plot_axis(axis_label, suffix):
    plt.figure(figsize=(12, 6))
    for bone in IMPORTANT_BONES:
        col = f"{bone}_{suffix}"
        if col in df.columns:
            plt.plot(t, df[col], label=bone)
    plt.xlabel("Time [s]")
    plt.ylabel(f"{axis_label} position")
    plt.title(f"OptiTrack main bones– {axis_label} over time")
    plt.legend(fontsize=8, ncol=3)
    plt.grid(True)
    plt.tight_layout()

# X, Y, Z plots
plot_axis("X", "X")
plot_axis("Y", "Y")
plot_axis("Z", "Z")

plt.show()