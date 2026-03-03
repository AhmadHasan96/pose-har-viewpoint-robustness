import os
import pandas as pd

RAW_DIR = "data/raw_landmarks"
OUTPUT = "data/all_raw_features_dataset.csv"


def extract_label(filename: str):
    filename = filename.lower()
    if "lifting" in filename: return "lifting"
    if "squatting" in filename: return "squatting"
    if "overhead" in filename: return "overhead"
    return None


def extract_video_name(filename: str):
    """
    Example:
    lifting_front_normal_1_landmarks.csv
    → lifting_front_normal_1
    """
    return filename.replace(".csv", "")


def merge_raw_features():
    all_rows = []

    for file in os.listdir(RAW_DIR):
        if not file.endswith(".csv"):
            continue

        label = extract_label(file)
        if label is None:
            print(f"[SKIP] No activity label found in filename → {file}")
            continue

        video_name = extract_video_name(file)
        file_path = os.path.join(RAW_DIR, file)

        df = pd.read_csv(file_path)

        # Add metadata columns
        df["movement"] = label
        df["video_name"] = video_name

        all_rows.append(df)
        print(f"[OK] Loaded {file}: {len(df)} frames")

    # Merge all data
    merged = pd.concat(all_rows, ignore_index=True)

    # Put metadata columns first
    columns = ["movement", "video_name"] + [c for c in merged.columns if c not in ["movement", "video_name"]]
    merged = merged[columns]

    # Save result
    merged.to_csv(OUTPUT, index=False)
    print(f"\n[SAVED] {OUTPUT}")
    print(f"Total frames: {len(merged)}")


if __name__ == "__main__":
    merge_raw_features()
