import os
import pandas as pd

RAW_DIR = "data/raw_landmarks"
OUTPUT = "data/all_raw_features_dataset.csv"

def extract_label_from_filename(filename):
    filename = filename.lower()
    if "lifting" in filename:
        return "lifting"
    if "squatting" in filename:
        return "squatting"
    if "overhead" in filename:
        return "overhead"
    return None


def merge_raw_features():
    all_rows = []

    for file in os.listdir(RAW_DIR):
        if not file.endswith(".csv"):
            continue

        label = extract_label_from_filename(file)
        if label is None:
            print(f"[SKIP] No label found in filename → {file}")
            continue

        file_path = os.path.join(RAW_DIR, file)
        df = pd.read_csv(file_path)

        # Add movement label for every frame
        df["movement"] = label

        all_rows.append(df)
        print(f"[OK] Loaded {file}, frames={len(df)}")

    # Merge all
    merged = pd.concat(all_rows, ignore_index=True)

    # Move "movement" to first column
    cols = ["movement"] + [c for c in merged.columns if c != "movement"]
    merged = merged[cols]

    # Save the result
    merged.to_csv(OUTPUT, index=False)

    print(f"\n[SAVED] {OUTPUT}")
    print(f"Total frames: {len(merged)}")


if __name__ == "__main__":
    merge_raw_features()
