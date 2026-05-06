import numpy as np
import pandas as pd

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

mp_df = pd.read_csv("New_vidoes/20.04.2026/Mod Data/Lifting arms sideways_MP_mod.csv")
# 1. GET MEDIAPIPE DATA (Landmarks 23 and 24)
# Replace these with your actual X, Y, Z values from your MediaPipe CSV
mp_l_hip = mp_df[["x23", "y23", "z23"]].values
mp_r_hip = mp_df[["x24", "y24", "z24"]].values
mp_width = get_distance(mp_l_hip, mp_r_hip)
print(mp_width)

opti_df = pd.read_csv("New_vidoes/20.04.2026/Mod Data/Lifting arms sideways_Opti_ExtraBones_mod.csv",header=[3,5,6])
def get_marker(df, name):
    x = df[(name, 'Position', 'X')].values
    y = df[(name, 'Position', 'Y')].values
    z = df[(name, 'Position', 'Z')].values
    return np.stack([x, y, z], axis=1)

mp_time = mp_df["rel_time_sec"].values
opti_time = opti_df[('Unnamed: 1_level_0', 'Unnamed: 1_level_1','Time (Seconds)')].values

def interpolate_marker(marker):
    # out = np.zeros((len(mp_time), 2))
    out = np.zeros((len(mp_time), 3))
    # for i in range(2):
    for i in range(3):
        out[:, i] = np.interp(mp_time, opti_time, marker[:, i])
    return out
# 2. GET OPTITRACK DATA (Hip and Thigh)
# Since OptiTrack's 'Hip' is the center, we find the distance to the 
# LThigh bone and double it to get the 'full' hip width.
ot_hip_center = interpolate_marker(get_marker(opti_df, 'FullBody:Hip'))
ot_l_thigh = interpolate_marker(get_marker(opti_df, 'FullBody:LThigh'))
ot_width = get_distance(ot_hip_center, ot_l_thigh) * 2
print(ot_width)
# 3. CALCULATE THE SCALE FACTOR
scale_factor = ot_width / mp_width

print(f"Scale Factor: {scale_factor:.2f}")
print(f"To align, multiply your MediaPipe coordinates by {scale_factor:.2f}")
