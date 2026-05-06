import pandas as pd
import numpy as np

# ---------------------------
# LOAD MEDIAPIPE (centered & scaled)
# ---------------------------

mp_df = pd.read_csv("New_vidoes/08.04.2026/Analysis Results/Front/Idle/MP_Scaled_Idle_Front.csv")
# ---------------------------
# LOAD OPTITRACK (centered & scaled, same column names)
# ---------------------------

opti_df = pd.read_csv("New_vidoes/08.04.2026/Analysis Results/Front/Idle/Opti_Scaled_Idle_Front.csv")

# Make sure we only work on common coordinate columns (exclude time)
coord_cols = [c for c in mp_df.columns if c in opti_df.columns and c != "time"]

# ---------------------------
# ESTIMATE CONSTANT OFFSET (MP - Opti)
# ---------------------------
offsets = {}
for col in coord_cols:
    diff = mp_df[col] - opti_df[col]          # per-frame difference
    offsets[col] = diff.mean()                # scalar offset for this joint+axis

# Optional: inspect offsets
offset_df = pd.DataFrame.from_dict(offsets, orient="index", columns=["offset"])
# print(offset_df)

mp_df_lifting = pd.read_csv("New_vidoes/08.04.2026/Analysis Results/Front/Lifting/MP_Scaled_Lifting_Front.csv")
# ---------------------------
# APPLY OFFSET CORRECTION TO MEDIAPIPE
# ---------------------------
mp_df_corr = mp_df_lifting.copy()
for col in coord_cols:
    mp_df_corr[col] = mp_df_lifting[col] - offsets[col]

# ---------------------------
# SAVE CORRECTED MEDIAPIPE
# ---------------------------
mp_df_corr.to_csv(

    "New_vidoes/08.04.2026/Analysis Results/Front/Lifting/MP_Scaled_Lifting_Front_NoOffset.csv",

    index=False
)
print("Offset-corrected file saved ✅")