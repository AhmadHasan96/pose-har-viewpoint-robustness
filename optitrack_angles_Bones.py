import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# # MediaPipe CSV
mp_df = pd.read_csv("New_vidoes/20.04.2026/Mod Data/Lifting arms sideways_MP_mod.csv")


# OptiTrack CSV (multi-header)
opti_df = pd.read_csv("New_vidoes/20.04.2026/Mod Data/lifting arms sideways_Opti_ExtraBones_mod.csv",header=[3,5,6])


bones_needed = [
    'FullBody:Chest', 
    'FullBody:LShoulder', 'FullBody:RShoulder',
    'FullBody:LUArm', 'FullBody:RUArm',
    'FullBody:LFArm', 'FullBody:RFArm',
    'FullBody:Hip', 'FullBody:LThigh', 'FullBody:RThigh',
    'FullBody:LShin', 'FullBody:RShin'
]

def quat_cols(bone):
    return [
        (bone, 'Rotation', 'X'),
        (bone, 'Rotation', 'Y'),
        (bone, 'Rotation', 'Z'),
        (bone, 'Rotation', 'W'),
    ]

def drop_bad_quat_rows(df, bones):
    df_clean = df.copy()

    bad_mask = np.zeros(len(df_clean), dtype=bool)

    for bone in bones:
        cols = quat_cols(bone)
        q = df_clean.loc[:, cols].astype(float).to_numpy()

        # bad if any NaN
        nan_bad = np.isnan(q).any(axis=1)

        # bad if quaternion norm is zero (or nearly zero)
        norms = np.linalg.norm(np.nan_to_num(q, nan=0.0), axis=1)
        zero_bad = norms < 1e-8

        bad_mask |= nan_bad | zero_bad

    print("Rows before:", len(df_clean))
    print("Rows dropped:", bad_mask.sum())

    df_clean = df_clean.loc[~bad_mask].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

def get_bone_quat(df, name):
    q = df.loc[:, quat_cols(name)].astype(float).to_numpy()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q

def relative_euler(df, parent_bone, child_bone, order='xyz'):
    q_parent = get_bone_quat(df, parent_bone)
    q_child  = get_bone_quat(df, child_bone)

    R_parent = R.from_quat(q_parent)
    R_child  = R.from_quat(q_child)

    R_rel = R_parent.inv() * R_child
    return R_rel.as_euler(order, degrees=True)

def interpolate_angles(opti_angles, opti_time, mp_time):
    """
    opti_angles: array (N,) or (N, K) from OptiTrack
    opti_time:   array (N,) timestamps for OptiTrack
    mp_time:     array (M,) timestamps for MediaPipe
    returns:     Opti angles interpolated at MP timestamps
    """
    opti_angles = np.asarray(opti_angles)

    if opti_angles.ndim == 1:
        return np.interp(mp_time, opti_time, opti_angles)

    out = np.zeros((len(mp_time), opti_angles.shape[1]))
    for i in range(opti_angles.shape[1]):
        out[:, i] = np.interp(mp_time, opti_time, opti_angles[:, i])
    return out

# 1) clean dataframe first
opti_df_clean = drop_bad_quat_rows(opti_df, bones_needed)

# 2) compute angles on cleaned data
L_sh_rel = relative_euler(opti_df_clean, 'FullBody:Chest', 'FullBody:LUArm')
R_sh_rel = relative_euler(opti_df_clean, 'FullBody:Chest', 'FullBody:RUArm')
# L_sh_rel = relative_euler(opti_df_clean, 'FullBody:LShoulder', 'FullBody:LUArm')
# R_sh_rel = relative_euler(opti_df_clean, 'FullBody:RShoulder', 'FullBody:RUArm')

L_el_rel = relative_euler(opti_df_clean, 'FullBody:LUArm', 'FullBody:LFArm')
R_el_rel = relative_euler(opti_df_clean, 'FullBody:RUArm', 'FullBody:RFArm')

L_h_rel  = relative_euler(opti_df_clean, 'FullBody:Hip', 'FullBody:LThigh')
R_h_rel  = relative_euler(opti_df_clean, 'FullBody:Hip', 'FullBody:RThigh')

L_kn_rel = relative_euler(opti_df_clean, 'FullBody:LThigh', 'FullBody:LShin')
R_kn_rel = relative_euler(opti_df_clean, 'FullBody:RThigh', 'FullBody:RShin')

# IMPORTANT:
# You must choose which Euler component corresponds to your motion.
# Start with these guesses, then verify visually on a few frames.
L_sh_deg = 90 - np.abs(L_sh_rel[:, 2])   # sideways arm raise often appears on one component
R_sh_deg = 90 - np.abs(R_sh_rel[:, 2])

L_el_deg = 180 - np.abs(L_el_rel[:, 2])   # elbow flex/ext
R_el_deg = 180 - np.abs(R_el_rel[:, 2])

L_h_deg  = 180 - np.abs(L_h_rel[:, 2])    # hip flex/ext or abd/add depending on convention
R_h_deg  =  180 -np.abs(R_h_rel[:, 2])

L_kn_deg = 180 - np.abs(L_kn_rel[:, 1])   # knee flex/ext
R_kn_deg = 180 - np.abs(R_kn_rel[:, 1])

# ---------------------------
# 2. GET TIME
# ---------------------------
mp_time = mp_df["rel_time_sec"].values
opti_time = opti_df_clean[('Unnamed: 1_level_0', 'Unnamed: 1_level_1','Time (Seconds)')].values

L_sh_angle_interp = interpolate_angles(L_sh_deg, opti_time, mp_time)
R_sh_angle_interp = interpolate_angles(R_sh_deg, opti_time, mp_time)
L_el_angle_interp = interpolate_angles(L_el_deg, opti_time, mp_time)
R_el_angle_interp = interpolate_angles(R_el_deg, opti_time, mp_time)
L_h_angle_interp = interpolate_angles(L_h_deg, opti_time, mp_time)
R_h_angle_interp = interpolate_angles(R_h_deg, opti_time, mp_time)
L_kn_angle_interp = interpolate_angles(L_kn_deg, opti_time, mp_time)
R_kn_angle_interp = interpolate_angles(R_kn_deg, opti_time, mp_time)



# creating pd 
angles_df = pd.DataFrame({
    "time": mp_df["rel_time_sec"],
    "L_sh_deg": L_sh_angle_interp,
    "R_sh_deg": R_sh_angle_interp,
    "L_el_deg": L_el_angle_interp,
    "R_el_deg": R_el_angle_interp,
    "L_h_deg": L_h_angle_interp,
    "R_h_deg": R_h_angle_interp,
    "L_kn_deg": L_kn_angle_interp,
    "R_kn_deg": R_kn_angle_interp,
})

angles_df.to_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/Opti_sideways_ExtraBones.csv", index=False)
print("Done ✅")
