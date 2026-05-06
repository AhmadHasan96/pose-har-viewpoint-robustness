import pandas as pd
import numpy as np

# ---------------------------
# LOAD
# ---------------------------

mp_df = pd.read_csv("New_vidoes/08.04.2026/Front/08.04.2026 Lifting front MP_Modified.csv")
# ---------------------------
# GET SHOULDERS
# ---------------------------
L_sh = mp_df[["x11", "y11", "z11"]].values
R_sh = mp_df[["x12", "y12", "z12"]].values

L_el = mp_df[["x13", "y13", "z13"]].values
R_el = mp_df[["x14", "y14", "z14"]].values

L_wr = mp_df[["x15", "y15", "z15"]].values
R_wr = mp_df[["x16", "y16", "z16"]].values

L_h = mp_df[["x23", "y23", "z23"]].values
R_h = mp_df[["x24", "y24", "z24"]].values
# ---------------------------
# CENTERING
# ---------------------------
center = (L_sh + R_sh) / 2

L_sh_c = L_sh - center
R_sh_c = R_sh - center
L_el_c = L_el - center
R_el_c = R_el - center
L_wr_c = L_wr - center
R_wr_c = R_wr - center
L_h_c = L_h - center
R_h_c = R_h - center

# ---------------------------
# SCALING (NEW)
# ---------------------------
scale = np.linalg.norm(L_sh - R_sh, axis=1)

L_sh_s = L_sh_c / scale[:, np.newaxis]
R_sh_s = R_sh_c / scale[:, np.newaxis]
L_el_s = L_el_c / scale[:, np.newaxis]
R_el_s = R_el_c / scale[:, np.newaxis]
L_wr_s = L_wr_c / scale[:, np.newaxis]
R_wr_s = R_wr_c / scale[:, np.newaxis]
L_h_s = L_h_c / scale[:, np.newaxis]
R_h_s = R_h_c / scale[:, np.newaxis]

# ---------------------------
# SAVE ONLY SHOULDERS
# ---------------------------
out_df = pd.DataFrame({
    "time": mp_df["rel_time_sec"],

    "L_sh_x": L_sh_s[:,0], "L_sh_y": -L_sh_s[:,1], "L_sh_z": -L_sh_s[:,2],
    "R_sh_x": R_sh_s[:,0], "R_sh_y": -R_sh_s[:,1], "R_sh_z": -R_sh_s[:,2],

    "L_el_x": L_el_s[:,0], "L_el_y": -L_el_s[:,1], "L_el_z": -L_el_s[:,2],
    "R_el_x": R_el_s[:,0], "R_el_y": -R_el_s[:,1], "R_el_z": -R_el_s[:,2],

    "L_wr_x": L_wr_s[:,0], "L_wr_y": -L_wr_s[:,1], "L_wr_z": -L_wr_s[:,2],
    "R_wr_x": R_wr_s[:,0], "R_wr_y": -R_wr_s[:,1], "R_wr_z": -R_wr_s[:,2],

    "L_h_x": L_h_s[:,0], "L_h_y": -L_h_s[:,1], "L_h_z": L_h_s[:,2],
    "R_h_x": R_h_s[:,0], "R_h_y": -R_h_s[:,1], "R_h_z": R_h_s[:,2],
})




out_df.to_csv("New_vidoes/08.04.2026/Analysis Results/Front/Lifting/MP_Scaled_Lifting_Front.csv", index=False)
print("Done ✅")