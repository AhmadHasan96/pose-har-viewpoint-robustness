import pandas as pd
import numpy as np

# ========= FILES =========
# MP_CSV = "New_videos/04.05.2026/Human Movement seperated/BendingForward/MediaPipe_BendingForward.csv"
# OPTI_CSV = "New_videos/04.05.2026/Human Movement seperated/BendingForward/OptiTrack_BendingForward.csv" 
# overall_MPJPE_OutputFile = "New_videos/04.05.2026/Human Movement seperated/BendingForward/overall_metrics.csv"
# PerJoint_MPJPE_OutputFile = "New_videos/04.05.2026/Human Movement seperated/BendingForward/per_joint_mpjpe.csv"
# PerJointAxis_MPJPE_OutputFile = "New_videos/04.05.2026/Human Movement seperated/BendingForward/per_joint_axis_mae.csv"
MP_CSV = "New_videos/04.05.2026/MediaPipe_main_joints_named.csv"
OPTI_CSV = "New_videos/04.05.2026/optitrack_NoOffset_main_bones_xyz.csv" 
overall_MPJPE_OutputFile = "New_videos/04.05.2026/overall_metrics.csv"
PerJoint_MPJPE_OutputFile = "New_videos/04.05.2026/per_joint_mpjpe.csv"
PerJointAxis_MPJPE_OutputFile = "New_videos/04.05.2026/per_joint_axis_mae.csv"

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

per_joint_mpjpe = {}
per_joint_axis_mae = {}

all_joint_distances = []
all_dx = []
all_dy = []
all_dz = []

for joint_name, (mp_name, opti_name) in joint_map.items():
    mp_xyz = mp[[f"{mp_name}_x", f"{mp_name}_y", f"{mp_name}_z"]].to_numpy(dtype=float)
    opti_xyz = opti[[f"{opti_name}_X", f"{opti_name}_Y", f"{opti_name}_Z"]].to_numpy(dtype=float)

    diff = mp_xyz - opti_xyz

    # keep only rows where both MP and Opti are valid
    mask = ~np.isnan(diff).any(axis=1)
    diff = diff[mask]

    if diff.size == 0:
        print(joint_name, "has no valid frames, skipping")
        continue

    dx = np.abs(diff[:, 0])
    dy = np.abs(diff[:, 1])
    dz = np.abs(diff[:, 2])

    dist = np.linalg.norm(diff, axis=1)

    per_joint_mpjpe[joint_name] = dist.mean()*100.0
    per_joint_axis_mae[joint_name] = {
        "x_mae": dx.mean()*100.0,
        "y_mae": dy.mean()*100.0,
        "z_mae": dz.mean()*100.0,
    }

    all_joint_distances.append(dist)
    all_dx.append(dx)
    all_dy.append(dy)
    all_dz.append(dz)
# ========= OVERALL METRICS =========
all_joint_distances = np.concatenate(all_joint_distances)
all_dx = np.concatenate(all_dx)
all_dy = np.concatenate(all_dy)
all_dz = np.concatenate(all_dz)

overall_mpjpe = all_joint_distances.mean()*100.0
overall_x_mae = all_dx.mean()*100.0
overall_y_mae = all_dy.mean()*100.0
overall_z_mae = all_dz.mean()*100.0

# ========= SAVE RESULTS TO CSV =========
overall_df = pd.DataFrame([{
    "overall_mpjpe": overall_mpjpe,
    "overall_x_mae": overall_x_mae,
    "overall_y_mae": overall_y_mae,
    "overall_z_mae": overall_z_mae,
}])

per_joint_df = pd.DataFrame([
    {"joint": joint, "mpjpe": value}
    for joint, value in per_joint_mpjpe.items()
])

per_joint_axis_df = pd.DataFrame([
    {
        "joint": joint,
        "x_mae": vals["x_mae"],
        "y_mae": vals["y_mae"],
        "z_mae": vals["z_mae"],
    }
    for joint, vals in per_joint_axis_mae.items()
])

overall_df.to_csv(overall_MPJPE_OutputFile, index=False)
per_joint_df.to_csv(PerJoint_MPJPE_OutputFile, index=False)
per_joint_axis_df.to_csv(PerJointAxis_MPJPE_OutputFile, index=False)

print("Saved: overall_metrics.csv")
print("Saved: per_joint_mpjpe.csv")
print("Saved: per_joint_axis_mae.csv")

print("\n=== OVERALL ===")
print(f"Overall MPJPE: {overall_mpjpe:.6f}")
print(f"Overall X MAE: {overall_x_mae:.6f}")
print(f"Overall Y MAE: {overall_y_mae:.6f}")
print(f"Overall Z MAE: {overall_z_mae:.6f}")

print("\n=== PER JOINT MPJPE ===")
for joint, value in per_joint_mpjpe.items():
    print(f"{joint}: {value:.6f}")

print("\n=== PER JOINT PER AXIS MAE ===")
for joint, vals in per_joint_axis_mae.items():
    print(
        f"{joint}: "
        f"X={vals['x_mae']:.6f}, "
        f"Y={vals['y_mae']:.6f}, "
        f"Z={vals['z_mae']:.6f}"
    )