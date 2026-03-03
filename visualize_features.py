import pandas as pd
import matplotlib.pyplot as plt
import os

# === SETTINGS ===
CSV_PATH = "data/features/squatting_side_normal_3_landmarks_features.csv" # e.g., data/features/lifting_front_normal_1_landmarks_features.csv
video_name = os.path.splitext(os.path.basename(CSV_PATH))[0]

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} frames with {len(df.columns)} features")

# === PLOT EACH FEATURE SEPARATELY ===
for col in df.columns:
    plt.figure(figsize=(8, 4))
    plt.plot(df[col], color='black', linewidth=1.5)
    plt.title(f"{video_name} — {col}")
    plt.xlabel("Frame")
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
