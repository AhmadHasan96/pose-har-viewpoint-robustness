import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------
# 1. LOAD FILES
# --------------------------------------------------
mp_df = pd.read_csv("New_vidoes/20.04.2026/lifting arms sideways_MP_mod.csv")

opti_df = pd.read_csv(
    "New_vidoes/20.04.2026/lifting arms sideways_Opti_ExtraBones.csv",
    header=[3, 5, 6]
)

# --------------------------------------------------
# 2. USE BETTER BONE SET
# --------------------------------------------------
bones_needed = [
    'FullBody:Hip',
    'FullBody:LShoulder', 'FullBody:RShoulder',
    'FullBody:LUArm', 'FullBody:RUArm',
    'FullBody:LFArm', 'FullBody:RFArm',
    'FullBody:LThigh', 'FullBody:RThigh',
    'FullBody:LShin', 'FullBody:RShin'
]

# --------------------------------------------------
# 3. HELPERS
# --------------------------------------------------
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

        nan_bad = np.isnan(q).any(axis=1)
        norms = np.linalg.norm(np.nan_to_num(q, nan=0.0), axis=1)
        zero_bad = norms < 1e-8

        bad_mask |= nan_bad | zero_bad

    print("Rows before:", len(df_clean))
    print("Rows dropped:", bad_mask.sum())

    df_clean = df_clean.loc[~bad_mask].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

def get_bone_quat(df, bone):
    q = df.loc[:, quat_cols(bone)].astype(float).to_numpy()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q

def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def angle_between_segments(u, v):
    u = normalize(u)
    v = normalize(v)
    c = np.sum(u * v, axis=-1)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def rotate_local_axis_to_global(q_xyzw, local_axis):
    q_xyzw = np.asarray(q_xyzw, dtype=float)
    local_axis = np.asarray(local_axis, dtype=float)

    rot = R.from_quat(q_xyzw)  # assumes XYZW order
    global_axis = rot.apply(np.tile(local_axis, (len(q_xyzw), 1)))
    return normalize(global_axis)

def interpolate_angles(opti_angles, opti_time, mp_time):
    opti_angles = np.asarray(opti_angles)

    if opti_angles.ndim == 1:
        return np.interp(mp_time, opti_time, opti_angles)

    out = np.zeros((len(mp_time), opti_angles.shape[1]))
    for i in range(opti_angles.shape[1]):
        out[:, i] = np.interp(mp_time, opti_time, opti_angles[:, i])
    return out

# --------------------------------------------------
# 4. CLEAN DATA
# --------------------------------------------------
opti_df_clean = drop_bad_quat_rows(opti_df, bones_needed)

# --------------------------------------------------
# 5. GET QUATERNIONS
# --------------------------------------------------
Hip_q       = get_bone_quat(opti_df_clean, 'FullBody:Hip')

LShoulder_q = get_bone_quat(opti_df_clean, 'FullBody:LShoulder')
RShoulder_q = get_bone_quat(opti_df_clean, 'FullBody:RShoulder')

LUArm_q     = get_bone_quat(opti_df_clean, 'FullBody:LUArm')
RUArm_q     = get_bone_quat(opti_df_clean, 'FullBody:RUArm')

LFArm_q     = get_bone_quat(opti_df_clean, 'FullBody:LFArm')
RFArm_q     = get_bone_quat(opti_df_clean, 'FullBody:RFArm')

LThigh_q    = get_bone_quat(opti_df_clean, 'FullBody:LThigh')
RThigh_q    = get_bone_quat(opti_df_clean, 'FullBody:RThigh')

LShin_q     = get_bone_quat(opti_df_clean, 'FullBody:LShin')
RShin_q     = get_bone_quat(opti_df_clean, 'FullBody:RShin')

# --------------------------------------------------
# 6. CANDIDATE LOCAL AXES
# --------------------------------------------------
candidate_axes = [
    np.array([ 1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([ 0, 1, 0]),
    np.array([ 0,-1, 0]),
    np.array([ 0, 0, 1]),
    np.array([ 0, 0,-1]),
]

# --------------------------------------------------
# 7. AXIS SEARCH USING NEUTRAL STANDING FRAMES
# --------------------------------------------------
# Assumption:
# first frames are standing / neutral
neutral_n = 30
neutral_slice = slice(0, neutral_n)

def pick_best_axis_pair(parent_q, child_q, target_deg=180.0, neutral_slice=slice(0, 30)):
    best = None
    best_err = np.inf

    for a_parent in candidate_axes:
        parent_dir = rotate_local_axis_to_global(parent_q, a_parent)

        for a_child in candidate_axes:
            child_dir = rotate_local_axis_to_global(child_q, a_child)
            ang = angle_between_segments(parent_dir, child_dir)

            err = np.mean(np.abs(ang[neutral_slice] - target_deg))

            if err < best_err:
                best_err = err
                best = {
                    "parent_axis": a_parent.copy(),
                    "child_axis": a_child.copy(),
                    "neutral_mean": float(np.mean(ang[neutral_slice])),
                    "neutral_err": float(err),
                    "full_angle": ang.copy()
                }

    return best

# --------------------------------------------------
# 8. FIND BEST AXES FOR EACH JOINT
# --------------------------------------------------
# Elbow and knee should clearly be close to 180 in standing
L_el_best = pick_best_axis_pair(LUArm_q,  LFArm_q,  target_deg=180.0, neutral_slice=neutral_slice)
R_el_best = pick_best_axis_pair(RUArm_q,  RFArm_q,  target_deg=180.0, neutral_slice=neutral_slice)
L_kn_best = pick_best_axis_pair(LThigh_q, LShin_q,  target_deg=180.0, neutral_slice=neutral_slice)
R_kn_best = pick_best_axis_pair(RThigh_q, RShin_q,  target_deg=180.0, neutral_slice=neutral_slice)

# For shoulder and hip, using 180 in neutral is still a practical target here
# because you want the same style as MediaPipe straight-line angle
L_sh_best = pick_best_axis_pair(LShoulder_q, LUArm_q, target_deg=90.0, neutral_slice=neutral_slice)
R_sh_best = pick_best_axis_pair(RShoulder_q, RUArm_q, target_deg=90.0, neutral_slice=neutral_slice)
L_h_best  = pick_best_axis_pair(Hip_q,       LThigh_q, target_deg=180.0, neutral_slice=neutral_slice)
R_h_best  = pick_best_axis_pair(Hip_q,       RThigh_q, target_deg=180.0, neutral_slice=neutral_slice)

print("L shoulder:", L_sh_best["parent_axis"], L_sh_best["child_axis"], L_sh_best["neutral_mean"])
print("R shoulder:", R_sh_best["parent_axis"], R_sh_best["child_axis"], R_sh_best["neutral_mean"])
print("L elbow   :", L_el_best["parent_axis"], L_el_best["child_axis"], L_el_best["neutral_mean"])
print("R elbow   :", R_el_best["parent_axis"], R_el_best["child_axis"], R_el_best["neutral_mean"])
print("L hip     :", L_h_best["parent_axis"],  L_h_best["child_axis"],  L_h_best["neutral_mean"])
print("R hip     :", R_h_best["parent_axis"],  R_h_best["child_axis"],  R_h_best["neutral_mean"])
print("L knee    :", L_kn_best["parent_axis"], L_kn_best["child_axis"], L_kn_best["neutral_mean"])
print("R knee    :", R_kn_best["parent_axis"], R_kn_best["child_axis"], R_kn_best["neutral_mean"])

# --------------------------------------------------
# 9. FINAL ANGLES
# --------------------------------------------------
L_sh_deg = L_sh_best["full_angle"]
R_sh_deg = R_sh_best["full_angle"]

L_el_deg = L_el_best["full_angle"]
R_el_deg = R_el_best["full_angle"]

L_h_deg  = L_h_best["full_angle"]
R_h_deg  = R_h_best["full_angle"]

L_kn_deg = L_kn_best["full_angle"]
R_kn_deg = R_kn_best["full_angle"]

# --------------------------------------------------
# 10. TIME
# --------------------------------------------------
mp_time = mp_df["rel_time_sec"].values

opti_time = opti_df_clean[
    ('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)')
].values

# --------------------------------------------------
# 11. INTERPOLATE TO MEDIAPIPE TIME
# --------------------------------------------------
L_sh_angle_interp = interpolate_angles(L_sh_deg, opti_time, mp_time)
R_sh_angle_interp = interpolate_angles(R_sh_deg, opti_time, mp_time)

L_el_angle_interp = interpolate_angles(L_el_deg, opti_time, mp_time)
R_el_angle_interp = interpolate_angles(R_el_deg, opti_time, mp_time)

L_h_angle_interp  = interpolate_angles(L_h_deg, opti_time, mp_time)
R_h_angle_interp  = interpolate_angles(R_h_deg, opti_time, mp_time)

L_kn_angle_interp = interpolate_angles(L_kn_deg, opti_time, mp_time)
R_kn_angle_interp = interpolate_angles(R_kn_deg, opti_time, mp_time)

# --------------------------------------------------
# 12. FINAL DATAFRAME
# --------------------------------------------------
angles_df = pd.DataFrame({
    "time": mp_df["rel_time_sec"],
    "L_sh_deg": L_sh_angle_interp,
    "R_sh_deg": R_sh_angle_interp,
    "L_el_deg": L_el_angle_interp,
    "R_el_deg": R_el_angle_interp,
    "L_h_deg":  L_h_angle_interp,
    "R_h_deg":  R_h_angle_interp,
    "L_kn_deg": L_kn_angle_interp,
    "R_kn_deg": R_kn_angle_interp,
})

# print(angles_df.head())

angles_df.to_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/Opti_ExtraBones_Mod.csv", index=False)
print("Done ✅")
