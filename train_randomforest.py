import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# === SETTINGS ===
DATA_PATH = "data/all_features_dataset.csv"
MODEL_PATH = "models/random_forest_model_unshuffeled.pkl"
os.makedirs("models", exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["movement"])
y = df["movement"]

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= False)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True, stratify=y)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# === TRAIN MODEL ===
rf = RandomForestClassifier(
    n_estimators=200,          # number of trees
    max_depth=None,            # let trees expand fully
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)
rf.fit(X_train, y_train)

# === EVALUATE ===
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === FEATURE IMPORTANCE ===
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(x=importances[:10], y=importances.index[:10], color="black")
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# === SAVE MODEL ===
joblib.dump(rf, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
