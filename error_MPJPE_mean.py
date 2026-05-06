import pandas as pd
import numpy as np

# ---------------------------
# LOAD
# ---------------------------

mp_df = pd.read_csv("New_vidoes/08.04.2026/Analysis Results/Front/Idle/Hip scaled/MP_Scaled_Idle_Front.csv")
# mp_df = pd.read_csv("New_vidoes/08.04.2026/Analysis Results/Front/MP_Scaled_Idle_Front.csv")


opti_df = pd.read_csv("New_vidoes/08.04.2026/Analysis Results/Front/Idle/Hip scaled/Opti_Scaled_Idle_Front.csv")


joints = ["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr", "L_h", "R_h", "L_kn", "R_kn"]

mpjpe_per_joint = {}
rows = []
for j in joints:
    dx = mp_df[f"{j}_x"] - opti_df[f"{j}_x"]
    dy = mp_df[f"{j}_y"] - opti_df[f"{j}_y"]
    dz = mp_df[f"{j}_z"] - opti_df[f"{j}_z"]

    # dist = np.sqrt(dx**2 + dy**2 + dz**2)      # per-frame 3D error
    # mpjpe_per_joint[j] = dist.mean()           # scalar MPJPE for this joint
    dist = np.sqrt(dx**2 + dy**2)  #2D
    rows.append({
        "joint": j,
        "mean_mpjpe": dist.mean(),
        "std_mpjpe": dist.std(),
        "max_mpjpe": dist.max()
    })

# mpjpe_df = pd.DataFrame.from_dict(mpjpe_per_joint, orient="index", columns=["MPJPE"])
mpjpe_df = pd.DataFrame.from_dict(rows)
# ---------------------------
# SAVE
# ---------------------------

# mpjpe_df.to_csv("New_vidoes/08.04.2026/Analysis Results/Front/Lifting/MPJPE Error/MPJPE_Lifting_WithZ_NoOffset.csv", index=False)
mpjpe_df.to_csv("New_vidoes/08.04.2026/Analysis Results/Front/Idle/Hip scaled/MPJPE Error/MPJPE_Idle_NoZ.csv", index=False)

print("Done ✅")