import pandas as pd
import matplotlib.pyplot as plt

# === SETTINGS ===
INPUT_CSV = "New_videos/04.05.2026/MP analysis/MediaPipe_main_joints_named.csv"

JOINTS = [
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

# === LOAD ===
df = pd.read_csv(INPUT_CSV)

t = df["time_ms"]

def plot_axis(axis_label):
    plt.figure(figsize=(12, 6))
    for joint in JOINTS:
        col = f"{joint}_{axis_label}"
        if col in df.columns:
            plt.plot(t, df[col], label=joint)

    plt.xlabel("Time [ms]")
    plt.ylabel(f"{axis_label.upper()} position")
    plt.title(f"MediaPipe main joints – {axis_label.upper()} over time")
    plt.legend(fontsize=8, ncol=3)
    plt.grid(True)
    plt.tight_layout()

# === PLOTS ===
plot_axis("x")
plot_axis("y")
plot_axis("z")

plt.show()