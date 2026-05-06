import numpy as np
import pandas as pd


mp_df = pd.read_csv("New_vidoes/20.04.2026/Mod Data/Lifting arms frontway_MP_mod.csv")

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

# L_sh = mp_df[["x11", "y11"]].values
# R_sh = mp_df[["x12", "y12"]].values

# L_el = mp_df[["x13", "y13"]].values
# R_el = mp_df[["x14", "y14"]].values

# L_wr = mp_df[["x15", "y15"]].values
# R_wr = mp_df[["x16", "y16"]].values

# L_h  = mp_df[["x23", "y23"]].values
# R_h  = mp_df[["x24", "y24"]].values

# L_kn = mp_df[["x25", "y25"]].values
# R_kn = mp_df[["x26", "y26"]].values

# L_heel = mp_df[["x29", "y29"]].values
# R_heel = mp_df[["x30", "y30"]].values


L_sh = mp_df[["x11", "y11", "z11"]].values
R_sh = mp_df[["x12", "y12", "z12"]].values

L_el = mp_df[["x13", "y13", "z13"]].values
R_el = mp_df[["x14", "y14", "z14"]].values

L_wr = mp_df[["x15", "y15", "z15"]].values
R_wr = mp_df[["x16", "y16", "z16"]].values

L_h  = mp_df[["x23", "y23", "z23"]].values
R_h  = mp_df[["x24", "y24", "z24"]].values

L_kn = mp_df[["x25", "y25", "z25"]].values
R_kn = mp_df[["x26", "y26", "z26"]].values

L_heel = mp_df[["x29", "y29", "z29"]].values
R_heel = mp_df[["x30", "y30", "z30"]].values

# Angle calcluation
L_sh_angle = angle_ABC(L_h, L_sh, L_el)
R_sh_angle = angle_ABC(R_h, R_sh, R_el)

L_el_angle = angle_ABC(L_sh, L_el, L_wr)
R_el_angle = angle_ABC(R_sh, R_el, R_wr)

L_h_angle = angle_ABC(L_sh, L_h, L_kn)
R_h_angle = angle_ABC(R_sh, R_h, R_kn)

L_kn_angle = angle_ABC(L_h, L_kn, L_heel)
R_kn_angle = angle_ABC(R_h, R_kn, R_heel)

# creating pd 
angles_df = pd.DataFrame({
    "time": mp_df["rel_time_sec"],
    "L_sh_deg": L_sh_angle,
    "R_sh_deg": R_sh_angle,
    "L_el_deg": L_el_angle,
    "R_el_deg": R_el_angle,
    "L_h_deg": L_h_angle,
    "R_h_deg": R_h_angle,
    "L_kn_deg": L_kn_angle,
    "R_kn_deg": R_kn_angle,
})

angles_df.to_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/MP_frontways_3D.csv", index=False)
print("Done ✅")
