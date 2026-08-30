import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ===== SETTINGS =====
angle = 360
level = "Chest"
model_complexity = 1
device_name = "OptiTrack"
REP_DIR = f"New_videos/August/Output/{device_name} Angles Calculation_{model_complexity}/{level}_Opti_4angles_3D_segment/reps_segment_{angle}°"   
TIME_COL = "rep_time"
SIG_COLs = ["L_shoulder_deg", "R_shoulder_deg", "L_hip_deg", "R_hip_deg"]  

N_POINTS = 101                                      # 0..100%
# ====================

rep_files = sorted(glob.glob(os.path.join(REP_DIR, "rep_*.csv")))



for SIG_COL in SIG_COLs:
    all_curves = []
    cycle_percent = np.linspace(0, 100, N_POINTS)

    plt.figure(figsize=(14, 6))
    for f in rep_files:
        df = pd.read_csv(f)
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
        df[SIG_COL] = pd.to_numeric(df[SIG_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL, SIG_COL]).reset_index(drop=True)

        t = df[TIME_COL].values
        y = df[SIG_COL].values

        if len(t) < 2 or t[-1] == t[0]:
            continue

        t_norm = (t - t[0]) / (t[-1] - t[0]) * 100.0
        y_norm = np.interp(cycle_percent, t_norm, y)

        all_curves.append(y_norm)
        plt.plot(cycle_percent, y_norm, color="gray", alpha=0.35)

    all_curves = np.array(all_curves)

    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    plt.plot(cycle_percent, mean_curve, color="blue", linewidth=3, label="Mean")
    plt.fill_between(
        cycle_percent,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="blue",
        alpha=0.2,
        label="Mean ± STD"
    )

    plt.xlabel("Cycle (%)")
    plt.ylabel(SIG_COL + " [°]")
    plt.title("Time-normalized repetitions (0–100%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(REP_DIR, f"{SIG_COL}_normalized.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    # plt.show()

    print(f"Used {len(all_curves)} repetitions")

    # save normalized reps
    all_reps_df = pd.DataFrame(
        all_curves.T,
        columns=[f"rep_{i+1:02d}" for i in range(len(all_curves))]
    )
    all_reps_df.insert(0, "cycle_percent", cycle_percent)
    all_reps_df.to_csv(os.path.join(REP_DIR, f"{SIG_COL}_normalized.csv"), index=False)

    # save mean/std
    summary_df = pd.DataFrame({
        "cycle_percent": cycle_percent,
        "mean": mean_curve,
        "std": std_curve,
        "mean_minus_std": mean_curve - std_curve,
        "mean_plus_std": mean_curve + std_curve
    })
    summary_df.to_csv(os.path.join(REP_DIR, f"{SIG_COL}_MeanStd.csv"), index=False)

    print("Saved normalized reps and mean/std CSVs")