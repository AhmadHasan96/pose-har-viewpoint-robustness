import pandas as pd
import numpy as np

# ---------------------------
# LOAD
# ---------------------------
mp_df = pd.read_csv("New_vidoes/Testing/Lifting_movement/mp_centered_scaled.csv")
opti_df = pd.read_csv("New_vidoes/Testing/Lifting_movement/opti_aligned_centered_scaled.csv")

# ---------------------------
# ALIGN LENGTH (safety)
# ---------------------------
min_len = min(len(mp_df), len(opti_df))

mp_df = mp_df.iloc[:min_len]
opti_df = opti_df.iloc[:min_len]

# ---------------------------
# FIND COMMON COLUMNS
# ---------------------------
common_cols = [c for c in mp_df.columns if c in opti_df.columns and c != "time"]

# ---------------------------
# COMPUTE ERRORS
# ---------------------------
error_data = {"time": mp_df["time"]}

for col in common_cols:
    error_data[col + "_error"] = np.abs(mp_df[col] - opti_df[col])

error_df = pd.DataFrame(error_data)

# ---------------------------
# SAVE
# ---------------------------
error_df.to_csv("New_vidoes/Testing/Lifting_movement/full_error.csv", index=False)

print("Done ✅")