import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

############################################### Segmentation ###########################################

# === LOAD SAVED MEDIAPIPE RAW CSV ===
Device = "OptiTrack"
level = "Chest"
Input_CSV = f"New_videos/August/Output/{Device} Angles Calculation_1/{level}_Opti_4angles_3D.csv"
SAVE_DIR = f"New_videos/August/Output/{Device} Angles Calculation_1"

df_mp_raw = pd.read_csv(Input_CSV)

print(f"✅ Loaded saved MediaPipe raw CSV → {Input_CSV}")
print(df_mp_raw.shape)
df_mp_raw.head()

# === MP SEGMENTATION SETTINGS ===
csv_name = os.path.splitext(os.path.basename(Input_CSV))[0]
SEGMENTS_DIR = os.path.join(SAVE_DIR, f"{csv_name}_segment")
os.makedirs(SEGMENTS_DIR, exist_ok=True)

TIME_COL = "time_s"
# TIME_COL = "Time_Seconds"
# SIG_COL = "y_15"   # MediaPipe raw left wrist Y (landmark 15)
SIG_COL = "L_shoulder_deg"

MIN_TIME_BETWEEN_PEAKS = 1.0
MIN_SEGMENT_DURATION = 10.0  # seconds
MIN_PEAK_HEIGHT = 120
MIN_PEAK_PROMINENCE = 0.05
SMOOTH_WINDOW = 5

SKIP_AFTER_JUMP = 2.0
STOP_BEFORE_NEXT_JUMP = 2.0
fps = 60

# === PREP RAW SIGNAL FOR SEGMENTATION ===
mp_seg_df = df_mp_raw[[TIME_COL, SIG_COL]].copy()
mp_seg_df = mp_seg_df.dropna().reset_index(drop=True)

print(mp_seg_df.head())
print(f"Rows used for segmentation: {len(mp_seg_df)}")

# === DETECT JUMP PEAKS ON RAW y_15 ===
from scipy.signal import find_peaks

min_distance_frames = max(1, int(MIN_TIME_BETWEEN_PEAKS * fps))

peaks, peak_props = find_peaks(
    mp_seg_df[SIG_COL].values,
    distance=min_distance_frames,
    height=MIN_PEAK_HEIGHT,
    prominence=MIN_PEAK_PROMINENCE
)

peak_times = mp_seg_df.loc[peaks, TIME_COL].values
peak_vals = mp_seg_df.loc[peaks, SIG_COL].values

print(f"Detected peaks: {len(peaks)}")
for i, (pt, pv) in enumerate(zip(peak_times, peak_vals), start=1):
    print(f"Peak {i}: time = {pt:.3f}s, L_shoulder_deg = {pv:.3f}")

# === CUT AND SAVE RAW MP SEGMENTS ===
TEST_ANGLES = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "360°"]
saved_segments = []
saved_id = 0

for i in range(len(peak_times) - 1):
    start_t = peak_times[i] + SKIP_AFTER_JUMP
    end_t   = peak_times[i + 1] - STOP_BEFORE_NEXT_JUMP

    if end_t <= start_t or (end_t - start_t) < MIN_SEGMENT_DURATION:
        print(
        f"Skipping segment {i+1}: "
        f"duration = {end_t - start_t:.2f}s "
        f"(minimum is {MIN_SEGMENT_DURATION:.1f}s)"
        )
        continue

    seg_df = df_mp_raw[(df_mp_raw[TIME_COL] >= start_t) & (df_mp_raw[TIME_COL] <= end_t)].copy()

    if seg_df.empty:
        print(f"Skipping segment {i+1}: empty segment")
        continue

    saved_id +=1
    seg_name = f"segment_{TEST_ANGLES[saved_id-1]}.csv"
    seg_path = os.path.join(SEGMENTS_DIR, seg_name)
    seg_df.to_csv(seg_path, index=False)

    
    saved_segments.append({
        "segment_id": saved_id,
        "start_s": start_t,
        "end_s": end_t,
        "n_frames": len(seg_df),
        "file": seg_name
    })

    print(f"Saved segment {i+1}: {seg_name} | {start_t:.2f}s -> {end_t:.2f}s | frames={len(seg_df)}")

#     # === SAVE SEGMENT SUMMARY ===
segments_summary_df = pd.DataFrame(saved_segments)

# summary_csv_path = os.path.join(SAVE_DIR, f"{video_name}_mp_segments_summary.csv")
# segments_summary_df.to_csv(summary_csv_path, index=False)

# print(f"✅ Saved segment summary → {summary_csv_path}")
print(segments_summary_df)
# === PLOT PEAK DETECTION AND SAVED SEGMENTS ===
fig, ax = plt.subplots(figsize=(16, 6))

# Original signal used for peak detection
ax.plot(
    mp_seg_df[TIME_COL],
    mp_seg_df[SIG_COL],
    color="steelblue",
    linewidth=1.2,
    label="Left Shoulder Angle"
)

# Detected peaks
ax.scatter(
    peak_times,
    peak_vals,
    color="crimson",
    s=60,
    zorder=3,
    label="Detected peaks"
)

# Horizontal threshold used by find_peaks
ax.axhline(
    MIN_PEAK_HEIGHT,
    color="darkorange",
    linestyle="--",
    linewidth=1,
    label=f"Minimum peak height ({MIN_PEAK_HEIGHT})"
)

# Shade each valid exported segment
for segment in saved_segments:
    start_t = segment["start_s"]
    end_t = segment["end_s"]
    segment_id = segment["segment_id"]

    ax.axvspan(
        start_t,
        end_t,
        color="limegreen",
        alpha=0.22,
        label="Saved segment" if segment_id == 1 else None
    )

    ax.text(
        (start_t + end_t) / 2,
        mp_seg_df[SIG_COL].min(),
        f"Segment {segment_id}",
        ha="center",
        va="bottom",
        color="darkgreen",
        fontsize=9,
        fontweight="bold"
    )

# Optional: draw vertical lines at every detected peak
for j, peak_t in enumerate(peak_times):
    ax.axvline(
        peak_t,
        color="crimson",
        linestyle=":",
        alpha=0.6,
        linewidth=1,
        label="Peak time" if j == 0 else None
    )

ax.set_title(f"{Device}: Segmentation {level}")
ax.set_xlabel("Time [s]")
# ax.set_ylabel("Left wrist vertical coordinate (y_15)")
ax.set_ylabel("Left Shoulder Angle [°]")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
# save_path = os.path.join(SEGMENTS_DIR, "Angles_repetition_plot.png")
# print(save_path)
# plt.savefig(save_path)
plt.show()