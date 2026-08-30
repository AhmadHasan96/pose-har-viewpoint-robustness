import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import numpy as np

# ===== SETTINGS =====
angle = 360
level = "Chest"
model_complexity = 1
device_name = "OptiTrack"
INPUT_CSV = f"New_videos/August/Output/{device_name} Angles Calculation_{model_complexity}/{level}_Opti_4angles_3D_segment/segment_{angle}°.csv"
SAVE_DIR = f"New_videos/August/Output/{device_name} Angles Calculation_{model_complexity}/{level}_Opti_4angles_3D_segment"
folder_name = f"reps_segment_{angle}°"

OUTPUT_DIR = os.path.join(SAVE_DIR, folder_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_COL = "time_s"       
SIG_COL = "L_shoulder_deg"   

DURATION_TOL = 0.45
RANGE_TOL = 0.45
FULL_CYCLE_THRESH = 0.9 # 35%
MIN_PEAK_PROMINENCE = 15.0
MIN_PEAK_HEIGHT = 50.0
MIN_TIME_BETWEEN_REPS = 1.0
# ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load
df = pd.read_csv(INPUT_CSV)
df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
df[SIG_COL] = pd.to_numeric(df[SIG_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL, SIG_COL]).reset_index(drop=True)

t = df[TIME_COL]
y = df[SIG_COL]

# smooth
# y_smooth = y.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

# broad peak detection
dt = t.diff().median()
min_distance_frames = max(1, int(MIN_TIME_BETWEEN_REPS / dt))

peaks, _ = find_peaks(
    # y_smooth.values,
    y.values,
    distance=min_distance_frames,
    prominence=MIN_PEAK_PROMINENCE,
    height=MIN_PEAK_HEIGHT,
    
)

peak_times = t.iloc[peaks].tolist()
# peak_vals = y_smooth.iloc[peaks].tolist()
peak_vals = y.iloc[peaks].tolist()

# build candidate reps
candidates = []
for i in range(len(peaks) - 1):
    start_idx = peaks[i]
    end_idx = peaks[i + 1]

    rep_df = df.iloc[start_idx:end_idx + 1].copy()
    # rep_y = y_smooth.iloc[start_idx:end_idx + 1].values
    rep_y = y.iloc[start_idx:end_idx + 1].values
    rep_t = t.iloc[start_idx:end_idx + 1].values

    # peak1 = y_smooth.iloc[start_idx]
    # peak2 = y_smooth.iloc[end_idx]
    peak1 = y.iloc[start_idx]
    peak2 = y.iloc[end_idx]
    valley = rep_y.min()

    drop1 = peak1 - valley
    drop2 = peak2 - valley
    cycle_depth = min(drop1, drop2)

    duration = rep_t[-1] - rep_t[0]
    value_range = rep_y.max() - rep_y.min()

    candidates.append({
        "i": i + 1,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_t": rep_t[0],
        "end_t": rep_t[-1],
        "duration": duration,
        "range": value_range,
        "peak1": peak1,
        "peak2": peak2,
        "valley": valley,
        "cycle_depth": cycle_depth,
        "df": rep_df
    })

durations = np.array([c["duration"] for c in candidates])
ranges = np.array([c["range"] for c in candidates])

med_duration = np.median(durations)
med_range = np.median(ranges)

dur_low = med_duration * (1 - DURATION_TOL)
dur_high = med_duration * (1 + DURATION_TOL)

range_low = med_range * (1 - RANGE_TOL)
range_high = med_range * (1 + RANGE_TOL)

cycle_depth_min = FULL_CYCLE_THRESH * med_range

print(f"Detected {len(peak_times)} peaks")
print(f"Median rep duration = {med_duration:.3f} s")
print(f"Median rep range = {med_range:.3f}")
print(f"Min full-cycle depth = {cycle_depth_min:.3f}")

# plot
plt.figure(figsize=(14, 6))
plt.plot(t, y, label=f"{SIG_COL} raw", alpha=0.30)
# plt.plot(t, y_smooth, label=f"{SIG_COL} smoothed", color="blue", linewidth=2)
plt.plot(t, y, label=f"{SIG_COL} smoothed", color="blue", linewidth=2)
plt.scatter(peak_times, peak_vals, color="red", zorder=5, label="Detected peaks")

saved_count = 0

for c in candidates:
    keep_duration = dur_low <= c["duration"] <= dur_high
    keep_range = range_low <= c["range"] <= range_high
    keep_full_cycle = c["cycle_depth"] >= cycle_depth_min

    keep = keep_duration and keep_range and keep_full_cycle

    color = "green" if keep else "red"
    alpha = 0.14 if keep else 0.10
    plt.axvspan(c["start_t"], c["end_t"], color=color, alpha=alpha)

    label_y = y.max() if keep else y.min()
    tag = f"R{c['i']}"
    plt.text((c["start_t"] + c["end_t"]) / 2, label_y, tag, ha="center", va="bottom", fontsize=8)

    if keep:
        original_id = c["i"]

        rep_df = c["df"].copy()
        rep_df["rep_id"] = original_id
        rep_df["rep_label"] = f"R{original_id}"
        rep_df["rep_time"] = rep_df[TIME_COL] - rep_df[TIME_COL].iloc[0]

        out_csv = os.path.join(OUTPUT_DIR, f"rep_R{original_id:02d}.csv")
        rep_df.to_csv(out_csv, index=False)

        print(
            f"KEEP R{original_id}: {c['start_t']:.3f}->{c['end_t']:.3f} s | "
            f"dur={c['duration']:.3f} | range={c['range']:.3f} | "
            f"depth={c['cycle_depth']:.3f} | saved -> {out_csv}"
        )
    else:
        print(
            f"DROP R{c['i']}: {c['start_t']:.3f}->{c['end_t']:.3f} s | "
            f"dur={c['duration']:.3f} | range={c['range']:.3f} | "
            f"depth={c['cycle_depth']:.3f}"
        )
for pt in peak_times:
    plt.axvline(pt, color="red", linestyle="--", alpha=0.6)

plt.xlabel("Time [s]")
plt.ylabel(SIG_COL + " [°]" )
plt.title(f"{device_name}: {folder_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()