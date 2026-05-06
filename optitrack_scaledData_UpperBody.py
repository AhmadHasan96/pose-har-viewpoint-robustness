import pandas as pd
import numpy as np

# ---------------------------
# 1. LOAD FILES
# ---------------------------

# # MediaPipe CSV
mp_df = pd.read_csv("New_vidoes/08.04.2026/Front/08.04.2026 Lifting front MP_Modified.csv")


# OptiTrack CSV (multi-header)
opti_df = pd.read_csv("New_vidoes/08.04.2026/Front/08.04.2026 Lifting front Opti_Modified.csv",header=[3,5,6])


# ---------------------------
# 2. GET TIME
# ---------------------------

mp_time = mp_df["rel_time_sec"].values
opti_time = opti_df[('Unnamed: 1_level_0', 'Unnamed: 1_level_1','Time (Seconds)')].values
# opti_time = opti_df[('Name', 'Unnamed: 1_level_1','Time (Seconds)')].values


# ---------------------------
# GET MARKERS (LEFT SHOULDER)
# ---------------------------

def get_marker(df, name):
    x = df[(name, 'Position', 'X')].values
    y = df[(name, 'Position', 'Y')].values
    z = df[(name, 'Position', 'Z')].values
    return np.stack([x, y, z], axis=1)

# Left shoulder
L_top = get_marker(opti_df, 'FullBody:LShoulderTop')
L_back = get_marker(opti_df, 'FullBody:LShoulderBack')
L_sh = (L_top + L_back) / 2

# Right shoulder
R_top = get_marker(opti_df, 'FullBody:RShoulderTop')
R_back = get_marker(opti_df, 'FullBody:RShoulderBack')
R_sh = (R_top + R_back) / 2



# ---------------------------
# INTERPOLATION
# ---------------------------

L_sh_interp = np.zeros((len(mp_time), 3))
R_sh_interp = np.zeros((len(mp_time), 3))
for i in range(3):
    L_sh_interp[:, i] = np.interp(mp_time, opti_time, L_sh[:, i])
    R_sh_interp[:, i] = np.interp(mp_time, opti_time, R_sh[:, i])

# ---------------------------
# CENTERING
# ---------------------------
center = (L_sh_interp + R_sh_interp) / 2

L_sh_centered = L_sh_interp - center
R_sh_centered = R_sh_interp - center

scale = np.linalg.norm(L_sh_centered - R_sh_centered, axis=1) 

L_scaled = L_sh_centered / scale[:, np.newaxis]
R_scaled = R_sh_centered / scale[:, np.newaxis]


def interpolate_marker(marker):
    out = np.zeros((len(mp_time), 3))
    for i in range(3):
        out[:, i] = np.interp(mp_time, opti_time, marker[:, i])
    return out

# Left /Right Elbow
L_elbow = get_marker(opti_df, 'FullBody:LElbowOut')
R_elbow = get_marker(opti_df, 'FullBody:RElbowOut')
# L_elbow_in = get_marker(opti_df, 'UpperBody:LElbowOut')
# L_u_arm = get_marker(opti_df, 'UpperBody:LUArmHigh')
# L_elbow = (L_elbow_in + L_u_arm) / 2

# R_elbow_in = get_marker(opti_df, 'UpperBody:RElbowOut')
# R_u_arm = get_marker(opti_df, 'UpperBody:RUArmHigh')
# R_elbow = (R_elbow_in + R_u_arm) / 2

# # Left Wrist
L_wr_in = get_marker(opti_df, 'FullBody:LWristIn')
L_wr_out = get_marker(opti_df, 'FullBody:LWristOut')
L_wr = (L_wr_in + L_wr_out) / 2
# Right Wrist
R_wr_in = get_marker(opti_df, 'FullBody:RWristIn')
R_wr_out = get_marker(opti_df, 'FullBody:RWristOut')
R_wr = (R_wr_in + R_wr_out) / 2
# Left Waist / hip
L_h_back = get_marker(opti_df, 'FullBody:WaistLBack')
# L_h_front = get_marker(opti_df, 'UpperBody:WaistLFront')
# L_h = (L_h_back + L_h_front) / 2
L_h = L_h_back
# Right Waist / hip
R_h_back = get_marker(opti_df, 'FullBody:WaistRBack')
# R_h_front = get_marker(opti_df, 'UpperBody:WaistRFront')
# R_h = (R_h_back + R_h_front) / 2
R_h = R_h_back


# Interpolate
L_elbow_interp = interpolate_marker(L_elbow)
R_elbow_interp = interpolate_marker(R_elbow)
L_wr_interp = interpolate_marker(L_wr)
R_wr_interp = interpolate_marker(R_wr)
L_h_interp = interpolate_marker(L_h)
R_h_interp = interpolate_marker(R_h)

# center
L_elbow_centered = L_elbow_interp - center
R_elbow_centered = R_elbow_interp - center
L_wr_centered = L_wr_interp - center
R_wr_centered = R_wr_interp - center
L_h_centered = L_h_interp - center
R_h_centered = R_h_interp - center

# Scale
L_elbow_scaled = L_elbow_centered / scale[:, np.newaxis]
R_elbow_scaled = R_elbow_centered / scale[:, np.newaxis]
L_wr_scaled = L_wr_centered / scale[:, np.newaxis]
R_wr_scaled = R_wr_centered / scale[:, np.newaxis]
L_h_scaled = L_h_centered / scale[:, np.newaxis]
R_h_scaled = R_h_centered / scale[:, np.newaxis]


# ---------------------------
# SAVE
# ---------------------------

out_df = pd.DataFrame({
    "time": mp_time,
    "L_sh_x": L_scaled[:,0],
    "L_sh_y": L_scaled[:,1],
    "L_sh_z": L_scaled[:,2],
    "R_sh_x": R_scaled[:,0],
    "R_sh_y": R_scaled[:,1],
    "R_sh_z": R_scaled[:,2],
    "L_el_x": L_elbow_scaled[:,0],
    "L_el_y": L_elbow_scaled[:,1],
    "L_el_z": L_elbow_scaled[:,2],
    "R_el_x": R_elbow_scaled[:,0],
    "R_el_y": R_elbow_scaled[:,1],
    "R_el_z": R_elbow_scaled[:,2],
    
    "L_wr_x": L_wr_scaled[:,0],
    "L_wr_y": L_wr_scaled[:,1],
    "L_wr_z": L_wr_scaled[:,2],
    "R_wr_x": R_wr_scaled[:,0],
    "R_wr_y": R_wr_scaled[:,1],
    "R_wr_z": R_wr_scaled[:,2],

    "L_h_x": L_h_scaled[:,0],
    "L_h_y": L_h_scaled[:,1],
    "L_h_z": L_h_scaled[:,2],
    "R_h_x": R_h_scaled[:,0],
    "R_h_y": R_h_scaled[:,1],
    "R_h_z": R_h_scaled[:,2],
})

out_df.to_csv("New_vidoes/08.04.2026/Analysis Results/Front/Lifting/Opti_Scaled_Lifting_Front.csv", index=False)


print("Done ✅")
