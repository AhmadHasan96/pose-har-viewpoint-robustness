import pandas as pd
import numpy as np

# ---------------------------
# LOAD
# ---------------------------

mp_df = pd.read_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/MP/MP_sideways_2D.csv")
mp_df   = mp_df.astype(float)

opti_df = pd.read_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/Opti/Opti_sideways_ExtraBones.csv")
opti_df = opti_df.astype(float)

angles = ["L_sh_deg", "R_sh_deg", "L_el_deg", "R_el_deg", "L_h_deg", "R_h_deg", "L_kn_deg", "R_kn_deg"]

error_stats = []
shift_n = 0   # +1 = MP later, -1 = MP earlier
for a in angles:
    # difference in degrees, same timestamp index assumed
    
    diff = mp_df[a].shift(shift_n) - opti_df[a]
    


    rel  = np.abs(diff) / np.abs(opti_df[a])  # relative error (0–1, per frame)
    mean_rel_percent = rel.mean() * 100       # mean error in %
    # wrap to [-180, 180] to avoid 359 vs 1 degree issues
    diff = (diff + 180) % 360 - 180
    
    mae = np.mean(np.abs(diff))     # mean absolute angle error
    # rmse = np.sqrt(np.mean(diff**2)) # root-mean-square error

    error_stats.append({
        "angle": a,
        # "error %": mean_rel_percent,
        "MAE_deg": mae,
        # "RMSE_deg": rmse
    })

angle_err_df = pd.DataFrame(error_stats)
print(angle_err_df)
# ---------------------------
# SAVE
# ---------------------------


# angle_err_df.to_csv("New_vidoes/20.04.2026/Analysis Results/Shoulders/Angle scaled/Error/error_sideways__2DMP_3DOOpti.csv", index=False)

print("Done ✅")