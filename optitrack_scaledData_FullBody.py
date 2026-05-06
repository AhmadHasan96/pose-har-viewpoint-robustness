import pandas as pd
import numpy as np

# ---------------------------
# 1. LOAD FILES
# ---------------------------

# # MediaPipe CSV
mp_df = pd.read_csv("New_vidoes/08.04.2026/Front/08.04.2026 Idle front MP_Modified.csv")


# OptiTrack CSV (multi-header)
opti_df = pd.read_csv("New_vidoes/08.04.2026/Front/08.04.2026 Idle front Opti_Modified.csv",header=[3,5,6])


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

# Left hip
# L_top = get_marker(opti_df, 'FullBody:LShoulderTop')
L_back = get_marker(opti_df, 'FullBody:WaistLBack')
# L_sh = (L_top + L_back) / 2
L_h = L_back

# Right hip
# R_top = get_marker(opti_df, 'FullBody:RShoulderTop')
R_back = get_marker(opti_df, 'FullBody:WaistRBack')
# R_sh = (R_top + R_back) / 2
R_h = R_back



# ---------------------------
# INTERPOLATION
# ---------------------------

L_h_interp = np.zeros((len(mp_time), 3))
R_h_interp = np.zeros((len(mp_time), 3))
for i in range(3):
    L_h_interp[:, i] = np.interp(mp_time, opti_time, L_h[:, i])
    R_h_interp[:, i] = np.interp(mp_time, opti_time, R_h[:, i])

# ---------------------------
# CENTERING
# ---------------------------
center = (L_h_interp + R_h_interp) / 2

L_h_centered = L_h_interp - center
R_h_centered = R_h_interp - center

scale = np.linalg.norm(L_h_centered - R_h_centered, axis=1) 

L_h_scaled = L_h_centered / scale[:, np.newaxis]
R_h_scaled = R_h_centered / scale[:, np.newaxis]


def interpolate_marker(marker):
    out = np.zeros((len(mp_time), 3))
    for i in range(3):
        out[:, i] = np.interp(mp_time, opti_time, marker[:, i])
    return out

# Left shoulder
L_top = get_marker(opti_df, 'FullBody:LShoulderTop')
L_back = get_marker(opti_df, 'FullBody:LShoulderBack')
L_sh = (L_top + L_back) / 2

# Right shoulder
R_top = get_marker(opti_df, 'FullBody:RShoulderTop')
R_back = get_marker(opti_df, 'FullBody:RShoulderBack')
R_sh = (R_top + R_back) / 2

# Left /Right Elbow
L_elbow = get_marker(opti_df, 'FullBody:LElbowOut')
R_elbow = get_marker(opti_df, 'FullBody:RElbowOut')
# L_elbow_in = get_marker(opti_df, 'FullBody:LElbowOut')
# L_u_arm = get_marker(opti_df, 'FullBody:LUArmHigh')
# L_elbow = (L_elbow_in + L_u_arm) / 2

# R_elbow_in = get_marker(opti_df, 'FullBody:RElbowOut')
# R_u_arm = get_marker(opti_df, 'FullBody:RUArmHigh')
# R_elbow = (R_elbow_in + R_u_arm) / 2

# # Left Wrist
L_wr_in = get_marker(opti_df, 'FullBody:LWristIn')
L_wr_out = get_marker(opti_df, 'FullBody:LWristOut')
L_wr = (L_wr_in + L_wr_out) / 2
# Right Wrist
R_wr_in = get_marker(opti_df, 'FullBody:RWristIn')
R_wr_out = get_marker(opti_df, 'FullBody:RWristOut')
R_wr = (R_wr_in + R_wr_out) / 2

# # Left/ Right Heel 
# L_heel = get_marker(opti_df, 'FullBody:LHeel')
# R_heel = get_marker(opti_df, 'FullBody:RHeel')

# Left Knee
L_kn = get_marker(opti_df, 'FullBody:LKneeOut')

# Right Knee
R_kn = get_marker(opti_df, 'FullBody:RKneeOut')



# FullBody:RToeTip : has data, FullBody:LToeTip :No data
# FullBody:RToeOut : has data, FullBody:LToeOut : No data
# FullBody:RToeIn : No data, FullBody:LToeIn : has data
# FullBody:RThigh : has data, FullBody:LThigh : has data
# FullBody:RShin : has data, FullBody:LShin : has data
# FullBody:RKneeOut : has data, FullBody:LKneeOut : has data
# FullBody:RHeel : No data, FullBody:LHeel : has data
# FullBody:RAnkleOut : No data, FullBody:LAnkleOut : No data





# Interpolate
L_sh_interp = interpolate_marker(L_sh)
R_sh_interp = interpolate_marker(R_sh)
L_elbow_interp = interpolate_marker(L_elbow)
R_elbow_interp = interpolate_marker(R_elbow)
L_wr_interp = interpolate_marker(L_wr)
R_wr_interp = interpolate_marker(R_wr)
L_kn_interp = interpolate_marker(L_kn)
R_kn_interp = interpolate_marker(R_kn)
# L_heel_interp = interpolate_marker(L_heel)
# R_heel_interp = interpolate_marker(R_heel)


# center
L_sh_centered = L_sh_interp - center
R_sh_centered = R_sh_interp - center
L_elbow_centered = L_elbow_interp - center
R_elbow_centered = R_elbow_interp - center
L_wr_centered = L_wr_interp - center
R_wr_centered = R_wr_interp - center
L_kn_centered = L_kn_interp - center
R_kn_centered = R_kn_interp - center
# L_heel_centered = L_heel_interp - center
# R_heel_centered = R_heel_interp - center

# Scale
L_sh_scaled = L_sh_centered / scale[:, np.newaxis]
R_sh_scaled = R_sh_centered / scale[:, np.newaxis]
L_elbow_scaled = L_elbow_centered / scale[:, np.newaxis]
R_elbow_scaled = R_elbow_centered / scale[:, np.newaxis]
L_wr_scaled = L_wr_centered / scale[:, np.newaxis]
R_wr_scaled = R_wr_centered / scale[:, np.newaxis]
L_kn_scaled = L_kn_centered / scale[:, np.newaxis]
R_kn_scaled = R_kn_centered / scale[:, np.newaxis]
# L_heel_scaled = L_heel_centered / scale[:, np.newaxis]
# R_heel_scaled = R_heel_centered / scale[:, np.newaxis]


# ---------------------------
# SAVE
# ---------------------------

out_df = pd.DataFrame({
    "time": mp_time,
    "L_sh_x": L_sh_scaled[:,0],
    "L_sh_y": L_sh_scaled[:,1],
    "L_sh_z": L_sh_scaled[:,2],
    "R_sh_x": R_sh_scaled[:,0],
    "R_sh_y": R_sh_scaled[:,1],
    "R_sh_z": R_sh_scaled[:,2],
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

    "L_kn_x": L_kn_scaled[:,0],
    "L_kn_y": L_kn_scaled[:,1],
    "L_kn_z": L_kn_scaled[:,2],
    "R_kn_x": R_kn_scaled[:,0],
    "R_kn_y": R_kn_scaled[:,1],
    "R_kn_z": R_kn_scaled[:,2],

    # "L_heel_x": L_heel_scaled[:,0],
    # "L_heel_y": L_heel_scaled[:,1],
    # "L_heel_z": L_heel_scaled[:,2],
    # "R_heel_x": R_heel_scaled[:,0],
    # "R_heel_y": R_heel_scaled[:,1],
    # "R_heel_z": R_heel_scaled[:,2],
})

out_df.to_csv("New_vidoes/08.04.2026/Analysis Results/Front/Idle/Hip scaled/Opti_Scaled_Idle_Front.csv", index=False)


print("Done ✅")
