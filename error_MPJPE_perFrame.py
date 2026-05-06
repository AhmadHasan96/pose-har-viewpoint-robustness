import pandas as pd
import numpy as np

# ---------------------------
# LOAD
# ---------------------------
mp_df = pd.read_csv("New_vidoes/Testing/Lifting_movement/mp_centered_scaled.csv")
opti_df = pd.read_csv("New_vidoes/Testing/Lifting_movement/opti_aligned_centered_scaled.csv")


joints = ["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr", "L_h", "R_h"]

mpjpe_df = pd.DataFrame({"time": mp_df["time"]})
for j in joints:
    dx = mp_df[f"{j}_x"] - opti_df[f"{j}_x"]
    dy = mp_df[f"{j}_y"] - opti_df[f"{j}_y"]
    dz = mp_df[f"{j}_z"] - opti_df[f"{j}_z"]

    # mpjpe_df[f"{j}_mpjpe"] = np.sqrt(dx**2 + dy**2 + dz**2) # 3D
    mpjpe_df[f"{j}_mpjpe"] = np.sqrt(dx**2 + dy**2) #2D


# ---------------------------
# SAVE
# ---------------------------
mpjpe_df.to_csv("New_vidoes/Testing/Lifting_movement/MPJPE_perFrame_errorNoZ.csv", index=False)

print("Done ✅")