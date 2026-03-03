import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
#  CONFIG
# ======================================================
CSV_PATH = "data/all_raw_features_dataset.csv"
MODEL_PATH = "models/randomforest_raw_shuffeled.pkl"


def train_raw_randomforest():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    print(f"[INFO] Dataset loaded: {df.shape[0]} frames, {df.shape[1]-1} features")

    # # Split into features (X) and label (y)
    X = df.drop(columns=['movement', 'video_name'])
    y = df["movement"]
    # Separate features and labels
    

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        shuffle=True,
        # shuffle=False,
        random_state=42
    )

    print("[INFO] Training RandomForest...")
    model = RandomForestClassifier(
    n_estimators=200,          # number of trees
    max_depth=None,            # let trees expand fully
    random_state=42,
    n_jobs=-1                  # use all CPU cores
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    print("\n================= RESULTS =================")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("===========================================\n")
# === CONFUSION MATRIX ===
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[SAVED] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_raw_randomforest()
