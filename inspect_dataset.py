import pandas as pd

# === SETTINGS ===
DATASET_PATH = "data/all_features_dataset.csv"

# === LOAD DATA ===
df = pd.read_csv(DATASET_PATH)
print(f"✅ Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
print("\n📋 Columns:", list(df.columns))

# === CHECK LABEL DISTRIBUTION ===
print("\n🔍 Samples per movement:")
print(df["movement"].value_counts())

# === SHOW BASIC STATS ===
print("\n📈 Feature summary (first few columns):")
print(df.describe().iloc[:, :7])

# === CHECK FOR MISSING DATA ===
missing = df.isnull().sum().sum()
if missing > 0:
    print(f"\n⚠️ Missing values found: {missing}")
else:
    print("\n✅ No missing values detected")

# === OPTIONAL: QUICK VISUAL CHECK ===
try:
    import matplotlib.pyplot as plt
    df["movement"].value_counts().plot(kind='bar', color='black', alpha=0.7)
    plt.title("Samples per Movement")
    plt.xlabel("Movement")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib to see a bar chart of class distribution.")
