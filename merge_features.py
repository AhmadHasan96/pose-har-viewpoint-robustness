import pandas as pd
import os

# === SETTINGS ===
FEATURES_DIR = "data/features"
OUTPUT_PATH = "data/all_features_dataset.csv"

# === GATHER ALL FEATURE FILES ===
all_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith("_features.csv")]
print(f"Found {len(all_files)} feature files")

merged_data = []

for file in all_files:
    # Example filename: lifting_front_normal_1_landmarks_features.csv
    # Extract main movement label and video name
    parts = file.replace(".csv", "").split("_")
    
    label = parts[0]                     # "lifting"
    video_name = "_".join(parts[:-1])    # "lifting_front_normal_1_landmarks_features" → or without _features
    
    # Remove optional trailing "_landmarks_features" if present
    video_name = video_name.replace("_landmarks_features", "").replace("_features", "")
    
    # Load CSV
    file_path = os.path.join(FEATURES_DIR, file)
    df = pd.read_csv(file_path)
    
    # Add metadata columns
    df["movement"] = label
    df["video_name"] = video_name
    
    merged_data.append(df)

# === MERGE & SAVE ===
if merged_data:
    full_df = pd.concat(merged_data, ignore_index=True)
    full_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Merged dataset saved to {OUTPUT_PATH}")
    print(f"Total samples: {len(full_df)}")
    print(f"Columns: {list(full_df.columns)}")
    print(f"🧩 Unique videos found: {full_df['video_name'].nunique()}")
else:
    print("⚠️ No feature files found in data/features/")

