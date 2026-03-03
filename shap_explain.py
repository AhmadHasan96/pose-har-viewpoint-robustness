import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# === Load model ===
model_path = "models/randomforest_byIndex_SideView.pkl"
clf = joblib.load(model_path)
print("[OK] Loaded model:", model_path)

# === Load dataset ===
df = pd.read_csv("data/all_features_dataset.csv")  # ANGLES DATASET

# Prepare features
X = df.drop(columns=["movement", "video_name"])

# === SHAP explainer ===
explainer = shap.TreeExplainer(clf)

# Use a subset for faster computation
sample = X.sample(400, random_state=42)

shap_values = explainer.shap_values(sample)

# === GLOBAL FEATURE IMPORTANCE ===
shap.summary_plot(shap_values, sample, plot_type="bar")

# === FULL SUMMARY PLOT (optional) ===
# shap.summary_plot(shap_values, sample)
