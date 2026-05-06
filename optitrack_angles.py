import numpy as np
import pandas as pd


# # MediaPipe CSV
mp_df = pd.read_csv("New_vidoes/20.04.2026/lifting arms sideways_MP_mod.csv")


# OptiTrack CSV (multi-header)
opti_df = pd.read_csv("New_vidoes/20.04.2026/lifting arms sideways_Opti_mod.csv",header=[3,5,6])

# ---------------------------
# 2. GET TIME
# ---------------------------
mp_time = mp_df["rel_time_sec"].values
opti_time = opti_df[('Unnamed: 1_level_0', 'Unnamed: 1_level_1','Time (Seconds)')].values


def get_marker(df, name):
    x = df[(name, 'Position', 'X')].values
    y = df[(name, 'Position', 'Y')].values
    z = df[(name, 'Position', 'Z')].values
    return np.stack([x, y, z], axis=1)
    # return np.stack([x, y], axis=1)


# def interpolate_marker(marker):
#     # out = np.zeros((len(mp_time), 2))
#     out = np.zeros((len(mp_time), 3))
#     # for i in range(2):
#     for i in range(3):
#         out[:, i] = np.interp(mp_time, opti_time, marker[:, i])
#     return out
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

def angle_ABC(A, B, C):
    BA = A - B
    BC = C - B
    # cosine of angle
    cos_ang = np.sum(BA * BC, axis=1) / (
        np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)
    )
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    ang = np.arccos(cos_ang)      # radians
    return np.degrees(ang)        # degrees


# Left shoulder

# L_top = get_marker(opti_df, 'FullBody:LShoulderTop')
L_back = get_marker(opti_df, 'FullBody:LShoulderBack')
# L_sh = (L_top + L_back) / 2
L_sh = L_back

# Right shoulder
# R_top = get_marker(opti_df, 'FullBody:RShoulderTop')
R_back = get_marker(opti_df, 'FullBody:RShoulderBack')
# R_sh = (R_top + R_back) / 2
R_sh = R_back

# Left /Right Elbow
L_el = get_marker(opti_df, 'FullBody:LElbowOut')
R_el = get_marker(opti_df, 'FullBody:RElbowOut')
# L_elbow_in = get_marker(opti_df, 'FullBody:LElbowOut')
L_u_arm = get_marker(opti_df, 'FullBody:LUArmHigh')
# L_el = (L_elbow_in + L_u_arm) / 2

# R_elbow_in = get_marker(opti_df, 'FullBody:RElbowOut')
R_u_arm = get_marker(opti_df, 'FullBody:RUArmHigh')
# R_el = (R_elbow_in + R_u_arm) / 2

# # Left Wrist
L_wr_in = get_marker(opti_df, 'FullBody:LWristIn')
L_wr_out = get_marker(opti_df, 'FullBody:LWristOut')
L_wr = (L_wr_in + L_wr_out) / 2
# Right Wrist
R_wr_in = get_marker(opti_df, 'FullBody:RWristIn')
R_wr_out = get_marker(opti_df, 'FullBody:RWristOut')
R_wr = (R_wr_in + R_wr_out) / 2

# Left hip
L_front = get_marker(opti_df, 'FullBody:WaistLFront')
L_back = get_marker(opti_df, 'FullBody:WaistLBack')
L_h = (L_front + L_back) / 2
# L_h = L_back

# Right hip
R_front = get_marker(opti_df, 'FullBody:WaistRFront')
R_back = get_marker(opti_df, 'FullBody:WaistRBack')
R_h = (R_front + R_back) / 2
# R_h = R_back

# # # Left/ Right Heel 
# L_heel = get_marker(opti_df, 'FullBody:LHeel')
# R_heel = get_marker(opti_df, 'FullBody:RHeel')

# Left/ Right Knee
L_kn = get_marker(opti_df, 'FullBody:LKneeOut')
R_kn = get_marker(opti_df, 'FullBody:RKneeOut')

# # Left/ Right ToeOut 
L_toe = get_marker(opti_df, 'FullBody:LToeOut')
R_toe = get_marker(opti_df, 'FullBody:RToeOut')



# Interpolate
# L_sh_interp = interpolate_marker(L_sh)
# R_sh_interp = interpolate_marker(R_sh)
# L_el_interp = interpolate_marker(L_el)
# R_el_interp = interpolate_marker(R_el)
# L_wr_interp = interpolate_marker(L_wr)
# R_wr_interp = interpolate_marker(R_wr)
# L_h_interp = interpolate_marker(L_h)
# R_h_interp = interpolate_marker(R_h)
# L_kn_interp = interpolate_marker(L_kn)
# R_kn_interp = interpolate_marker(R_kn)
# L_heel_interp = interpolate_marker(L_heel)
# R_heel_interp = interpolate_marker(R_heel)

# # Angle calcluation
# L_sh_angle = angle_ABC(L_h_interp, L_sh_interp, L_el_interp)
# R_sh_angle = angle_ABC(R_h_interp, R_sh_interp, R_el_interp)

# L_el_angle = angle_ABC(L_sh_interp, L_el_interp, L_wr_interp)
# R_el_angle = angle_ABC(R_sh_interp, R_el_interp, R_wr_interp)

# L_h_angle = angle_ABC(L_sh_interp, L_h_interp, L_kn_interp)
# R_h_angle = angle_ABC(R_sh_interp, R_h_interp, R_kn_interp)

# L_kn_angle = angle_ABC(L_h_interp, L_kn_interp, L_heel_interp)
# R_kn_angle = angle_ABC(R_h_interp, R_kn_interp, R_heel_interp)

# Angle calcluation
L_sh_angle = angle_ABC(L_h, L_sh, L_el)
R_sh_angle = angle_ABC(R_h, R_sh, R_el)

L_el_angle = angle_ABC(L_u_arm, L_el, L_wr)
R_el_angle = angle_ABC(R_u_arm, R_el, R_wr)

L_h_angle = angle_ABC(L_sh, L_h, L_kn)
R_h_angle = angle_ABC(R_sh, R_h, R_kn)

L_kn_angle = angle_ABC(L_h, L_kn, L_toe)
R_kn_angle = angle_ABC(R_h, R_kn, R_toe)

L_sh_angle_interp = interpolate_angles(L_sh_angle, opti_time, mp_time)
R_sh_angle_interp = interpolate_angles(R_sh_angle, opti_time, mp_time)
L_el_angle_interp = interpolate_angles(L_el_angle, opti_time, mp_time)
R_el_angle_interp = interpolate_angles(R_el_angle, opti_time, mp_time)
L_h_angle_interp = interpolate_angles(L_h_angle, opti_time, mp_time)
R_h_angle_interp = interpolate_angles(R_h_angle, opti_time, mp_time)
L_kn_angle_interp = interpolate_angles(L_kn_angle, opti_time, mp_time)
R_kn_angle_interp = interpolate_angles(R_kn_angle, opti_time, mp_time)



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

angles_df.to_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/Opti_UArm.csv", index=False)
print("Done ✅")
