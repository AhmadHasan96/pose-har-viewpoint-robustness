import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== SETTINGS =====
angle = 90
level = "FeetView"
model_complexity = 1
# NameOfJoint = ["L_shoulder_deg","R_shoulder_deg", "L_hip_deg",  "R_hip_deg"]
NameOfJoint = ["R_shoulder_deg"]

for  joint in NameOfJoint:
    MP_SUMMARY = f"New_videos/July/14.07.2026/Output/MediaPipe Angles Calculation_{model_complexity}/{level}_MP_4angles_3D_segment/reps_segment_{angle}°/{joint}_MeanStd.csv"
    OPTI_SUMMARY = f"New_videos/July/14.07.2026/Output/OptiTrack Angles Calculation_{model_complexity}/{level}_OPT_4angles_3D_segment/reps_segment_{angle}°/{joint}_MeanStd.csv"
    OUT_DIR = f"New_videos/July/14.07.2026/{level} Comparision Output_{model_complexity}/{angle}°/{joint}"
    os.makedirs(OUT_DIR, exist_ok=True)
    # ====================

    mp = pd.read_csv(MP_SUMMARY)
    opti = pd.read_csv(OPTI_SUMMARY)

    # make sure numeric
    for col in ["cycle_percent", "mean", "std", "mean_minus_std", "mean_plus_std"]:
        mp[col] = pd.to_numeric(mp[col], errors="coerce")
        opti[col] = pd.to_numeric(opti[col], errors="coerce")

    # keep only valid rows
    mp = mp.dropna(subset=["cycle_percent", "mean", "std"]).reset_index(drop=True)
    opti = opti.dropna(subset=["cycle_percent", "mean", "std"]).reset_index(drop=True)

    # merge on cycle_percent
    df = pd.merge(mp, opti, on="cycle_percent", suffixes=("_mp", "_opti"))

    # compare
    df["mean_diff"] = df["mean_mp"] - df["mean_opti"]
    df["abs_mean_diff"] = np.abs(df["mean_diff"])
    df["std_diff"] = df["std_mp"] - df["std_opti"]
    df["abs_std_diff"] = np.abs(df["std_diff"])

    # global summary values
    mean_abs_diff_avg = df["abs_mean_diff"].mean()
    mean_abs_diff_max = df["abs_mean_diff"].max()
    std_abs_diff_avg = df["abs_std_diff"].mean()
    std_abs_diff_max = df["abs_std_diff"].max()

    summary_overall = pd.DataFrame([{
        "mean_abs_diff_avg": mean_abs_diff_avg,
        "mean_abs_diff_max": mean_abs_diff_max,
        "std_abs_diff_avg": std_abs_diff_avg,
        "std_abs_diff_max": std_abs_diff_max
    }])

    # save csvs
    df.to_csv(os.path.join(OUT_DIR, f"{joint}_mp_vs_opti_mean_std_comparison.csv"), index=False)
    summary_overall.to_csv(os.path.join(OUT_DIR, f"{joint}_mp_vs_opti_overall_summary.csv"), index=False)

    # ===== PLOT 1: means with std bands =====
    plt.figure(figsize=(12, 6))

    plt.plot(df["cycle_percent"], df["mean_mp"], label="MediaPipe mean", color="blue", linewidth=2.5)
    plt.fill_between(
        df["cycle_percent"],
        df["mean_minus_std_mp"],
        df["mean_plus_std_mp"],
        color="blue",
        alpha=0.18
    )

    plt.plot(df["cycle_percent"], df["mean_opti"], label="OptiTrack mean", color="red", linewidth=2.5)
    plt.fill_between(
        df["cycle_percent"],
        df["mean_minus_std_opti"],
        df["mean_plus_std_opti"],
        color="red",
        alpha=0.18
    )
    plt.xticks(fontsize=16 )
    plt.yticks(fontsize=16 )

    plt.xlabel("Cycle (%)", fontsize=16)
    plt.ylabel("Signal [°]", fontsize=16)
    plt.title("MediaPipe vs OptiTrack mean ± std",
    fontsize=14,
    )
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{joint}_mean_std_overlay.svg"), dpi=300)
    # plt.show()

    # ===== PLOT 2: absolute difference =====

 
    
    plt.figure(figsize=(12, 5))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(df["cycle_percent"], df["abs_mean_diff"], color="purple", linewidth=2.5, label="|Mean difference|")
    plt.xlabel("Cycle (%)", fontsize=16)
    plt.ylabel("Absolute difference [°]", fontsize=16)
    plt.title("Absolute difference between mean curves", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)

    metrics_text = (
        f"Mean abs diff = {mean_abs_diff_avg:.4f} °\n"
        f"Max abs diff = {mean_abs_diff_max:.4f} °"
    )

    plt.text(
        0.02, 0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{joint}_absolute_mean_difference_curve.svg"), dpi=300)
    # plt.show()

    print("Saved:")

    print(os.path.join(OUT_DIR, "mp_vs_opti_overall_summary.csv"))
