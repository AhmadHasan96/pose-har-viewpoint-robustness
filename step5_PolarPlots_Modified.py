import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

level = "ChestLevel"
joint_name = "L_shoulder_deg"
model_complex = 1
# save_dir = f"New_videos/August/{level} Comparision Output_{model_complex}"
save_dir = f"New_videos/July/14.07.2026/{level} Comparision Output_RP_{model_complex}"
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315, 360]

##################################### Feet Level ##########################################
# values_mean = [17.7, 21.64, 6.18, 8.61, 16.13, 14.29, 5.98, 10.38, 17.93] #  L_shoulder_deg
# values_std = [23.49, 9.27, 2.15, 0.72, 5.11 , 6.11 , 2.98, 2.24, 19.83]

# values_mean = [14.32, 18.56, 1.5, 9.1, 11.8, 5.77, 2.08, 9.58, 15.65] #  R_shoulder_deg
# values_std = [24.05, 8.85, 1.98, 2.04, 3.93 , 7.86 , 2.16, 1.86, 20.68]

# values_mean = [4.86, 9.42, 11.76, 4.37, 12.01, 6.19, 1.98, 7.39, 4.93] #  L_hip_deg
# values_std = [8.1, 0.96, 0.25, 1.36, 1.02 , 1.23 , 0.57, 0.59, 7.04]

# values_mean = [3.61, 2.03, 1.94, 8.7, 11.06, 0.73, 5.69, 2.34, 3.26] #  R_hip_deg
# values_std = [4.63, 2.21, 0.42, 0.64, 0.99 , 1.03 , 0.75, 0.95, 4.36]

##################################### Hip Level ##########################################
# values_mean = [11.05, 10.07, 9.64, 5.05, 13.12, 5.98, 14.62, 13.39, 9.08] #  L_shoulder_deg
# values_std = [5.22 ,4.81 , 1.73, 1.12 , 2.26, 2.19, 3.23, 1.97, 2.13]

# values_mean = [7.31, 14.87, 9.48, 4.18, 9.29, 6.37, 7.07,9.25, 6.25] #  R_shoulder_deg
# values_std = [6.6, 5.47, 2.93, 1.63, 3.93, 1.56, 1.86, 1.99, 2.17]

# values_mean = [8.35, 4.71, 9.36, 8.01, 3.57, 2.37,2.71, 9.07, 7.6] #  L_hip_deg
# values_std = [0.43, 0.96, 0.69, 0.36, 0.6, 0.45, 0.81, 0.72, 0.79]

# values_mean = [6.33, 2.3, 2.58, 2.77, 5.39, 3.97, 7.9, 2.41, 7.42] #  R_hip_deg
# values_std = [0.63, 1.01, 1.18, 0.33, 0.63, 0.48, 0.7, 0.55, 0.84]


##################################### Chest Level Faulty ##########################################
# values_mean = [7.14, 4.06, 10.07, 4.72, 15.14, 6.82, 10.21, 11.2, 5.45] #  L_shoulder_deg
# values_std = [2.13, 4.22, 1.03, 0.97 , 5.45, 4.16, 5.06, 2.46, 1.8]

# values_mean = [7.22, 7.62, 10.6, 6.79, 15, 13.6, 8.14, 5.34, 6.97] #  L_shoulder_deg_2
# values_std = [2.48, 1.23, 1.25, 1.63, 3.48, 2.76, 1.82, 2.48, 1.38]

# values_mean = [7.27, 9.82, 22.03, 7.13, 9.88, 6.77, 6.3, 7.71, 6.33] #  R_shoulder_deg
# values_std = [3.37, 4.91, 3.57, 1.19, 4.81, 2.76, 2.79, 2.81, 2.02]

# values_mean = [21.2, 19.3, 17.5, 4.53, 8.98, 8.79, 10.84, 15.12, 19.02 ] #  R_shoulder_deg_2
# values_std = [2.7, 0.98, 3.71, 1.21, 2.74, 1.71, 0.84, 1.84, 0.91]

# values_mean = [8.74, 4.44, 4.24, 7.68, 4.12, 2.6, 1.66, 5.72, 7.39] #  L_hip_deg
# values_std = [0.8, 0.88, 1.977, 0.38, 0.93, 0.68, 0.77, 0.45, 0.68]

# values_mean = [5, 3.59, 6.08, 8.5, 2.88, 4.37, 13.96, 10.22, 5.24] #  L_hip_deg_2
# values_std = [0.46, 0.33, 3.2, 0.19, 0.43, 0.38, 0.28, 0.39, 0.33]

# values_mean = [4.92, 2.73, 42.64, 4.28, 4.92, 6.48, 5.58,2.35, 4.68] #  R_hip_deg
# values_std = [1.03, 0.86, 19.69,0.52, 1.64, 1.25, 1.33, 0.84, 0.84]

# values_mean = [4.92, 10.24, 55.78, 17.93, 14.51, 5.12, 10.09, 7.84, 5.2] #  R_hip_deg_2
# values_std = [0.69, 0.73, 20.7, 0.82, 1.07, 1.31, 0.44, 0.59, 0.72]

##################################### Chest Level Repeated ##########################################
# values_mean = [7.1, 5.96, 5.58, 4.5, 9.46, 12.68, 8.25, 8.32, 4.73] #  L_shoulder_deg_0
# values_std = [1.98, 2.98, 1.88, 1.67, 3.1, 2.24, 2.75, 1.44, 1.91]

values_mean = [6.16, 4.11, 6.76, 5.37, 15.84, 10.41, 12.53, 11.11, 5.42] #  L_shoulder_deg
values_std = [1.59, 2.94, 1.24, 1.17, 2.84, 2.49, 3.13,1.22, 1.27]

# values_mean = [9.64, 9.44, 11.5, 9.3, 13.89, 12.29, 10.99, 4.71, 8.57] #  L_shoulder_deg_2
# values_std = [1.64, 1.84, 1.96, 1.39, 3.46, 2.86, 1.85, 0.96, 1.41]

# values_mean = [5.5, 13.2, 12.49, 15.01, 9.04, 6.61, 7.05, 7.88, 3.63] #  R_shoulder_deg_0
# values_std = [2.38, 3.82, 2.03, 1.89, 3.47, 1.98, 2.28, 1.73, 1.6]

# values_mean = [5.64, 10.44, 9.9, 4.45, 11.16, 6.51, 7.4, 7.52, 5.03] #  R_shoulder_deg
# values_std = [1.88, 3.43, 1.85, 1.64, 3.42, 1.81, 1.93, 1.55, 1.3]

# values_mean = [16.64, 18.66, 18.68, 5.61, 7.14, 7.97, 11.78, 12.57, 15.62] #  R_shoulder_deg_2
# values_std = [1.14, 1.21, 2.89, 0.82, 2.73, 1.96, 1.32, 0.78, 0.78]

# values_mean = [5.71, 5.49, 10.5, 3.18, 2.52, 3.32, 5.02, 11.82, 8.67] #  L_Hip_deg_0
# values_std = [0.49, 0.55, 0.42, 0.2, 0.29, 0.61, 0.2, 0.69, 0.63]

# values_mean = [9, 5.5, 11.46, 8.23, 3.12, 2.23, 1.72, 8.55, 13.5] #  L_Hip_deg
# values_std = [0.51, 0.89, 0.633, 0.38, 0.54, 0.65, 0.49, 0.46, 0.48]

# values_mean = [4.78, 4.02, 5.69, 7.66, 6.03, 2.58, 12.72, 10.09, 7.23] #  L_Hip_deg_2
# values_std = [0.61, 0.5, 0.44, 0.48, 0.33, 0.91, 0.38, 0.34, 0.6]

# values_mean = [7.62, 2.81, 3.43, 16.08, 3.05, 2.06, 4.79, 3.33, 8.08] #  R_hip_deg_0
# values_std = [1.8, 0.99, 0.66, 0.55, 0.59, 0.67, 0.34, 0.74, 0.78]

# values_mean = [8.5, 2.1, 2.08, 3.01, 2.92, 7.43, 5.14, 3.62, 8.49] #  R_hip_deg
# values_std = [1.47, 0.87, 0.89, 0.34, 0.71, 0.68, 0.69, 1.17, 0.72]

# values_mean = [8.01, 11.11, 13.14, 21.19, 5.78, 2.72, 12.59, 8.53, 7.67] #  R_hip_deg_2
# values_std = [1.53, 1.03, 0.72, 0.63, 0.48, 0.97, 0.55, 0.38, 0.86]

##################################### Chest Level Uniform Background ##########################################
# values_mean = [10.35, 5.27, 7.76, 7.87, 14.51, 7.62, 14.84, 12.25, 6.79] #  L_shoulder_deg
# values_std = [3.74, 4.54, 4.64, 5.34, 4.43, 7.69, 8.03, 8.59, 4.98]

# values_mean = [10.8, 9.31, 10.48, 7.57, 9.14, 4.45, 7.61, 6.48, 5.26] #  R_shoulder_deg
# values_std = [5.48, 4.39, 6.39, 4.41, 4.8, 6.1, 5.82, 7.12, 6.02]

# values_mean = [10.07, 5.04, 10.14] #  L_Hip_deg
# values_std = [1.02, 0.44, 0.61]

# values_mean = [4.81, 2.35, 4.58] #  R_hip_deg
# values_std = [0.78, 1.12, 1.13]

##################################### Head Level ##########################################
# values_mean = [8.18, 2.87, 10.69, 7.7, 17.1, 11.35, 12.67, 9.46, 5.99] #  L_shoulder_deg
# values_std = [ 2.87, 1.01 ,1.45 , 1.52, 5.02, 4.12, 2.52 , 1.66, 2.13]

# values_mean = [6.05, 9.73, 14.07, 11.25, 14.31, 16.7, 8.93, 5.5, 8.05] #  L_shoulder_deg_2
# values_std = [1.77, 1.46, 1.08, 1.87, 3.92, 2.79, 2.07, 1.13, 1.96]

# values_mean = [3.66, 7.66, 9.28, 5.38, 9, 4.17, 6.13, 7.88, 3.58] #  R_shoulder_deg
# values_std = [3.19, 0.84, 2.43, 1.69, 4.77, 0.79, 1.22, 1.89, 2.1]

# values_mean = [13.61, 18.45, 13.21, 3.47, 4.59, 8.97, 9.63, 13.87, 13.07] #  R_shoulder_deg_2
# values_std = [0.91, 0.88, 1.12, 1.27, 2.4, 2.01, 1.01, 0.96, 0.95]

# values_mean = [8.13, 5.01, 9.88, 7.4, 2.7, 2.33, 3.2, 5.65,7.36 ] #  L_hip_deg
# values_std = [0.88, 0.53, 1.07, 0.27, 1.14, 0.65, 0.72, 0.61, 0.83]

# values_mean = [] #  L_hip_deg_2
# values_std = []

# values_mean = [4.09, 3.42, 5.38, 0.91, 4.17, 3.12, 1.47, 2.31, 4.44 ] #  R_hip_deg
# values_std = [0.97, 0.44, 4.89, 0.6, 0.799, 0.355, 1.24, 1.09, 0.98]

# values_mean = [] #  R_hip_deg_2
# values_std = []

##################################### Bird Level ##########################################
# values_mean = [10.08, 8.44, 12.35, 12.69, 26.2, 20.44, 10.64, 9.26, 12.64] #  L_shoulder_deg
# values_std = [ 1.46,1.46 , 1.89, 4.12, 3.18, 1.81,4.61 ,1.32 ,2.62 ]

# values_mean = [8.05, 8.04, 12.79, 5.99, 16.33, 10.08, 7.64, 13.32, 9.3] #  R_shoulder_deg
# values_std = [1.62, 1.84, 2.47, 4.09, 3.28, 1.38, 3.13, 1.55, 3.17]

# values_mean = [5.73, 4.44, 13.75, 5.35, 1.85, 3.93, 2.95, 4.85, 4.5] #  L_Hip_deg
# values_std = [2.11, 1.12, 1.13, 0.59, 0.69, 0.78, 1.61, 1.08, 3.53]

# values_mean = [4.43, 4.33, 2.08, 8.34, 5.72, 1.03, 9.11, 6.18, 3.34] #  R_hip_deg
# values_std = [1.38, 1.09, 1.15, 1.16, 1.22, 0.71, 1.31, 0.47, 2.36]

# Convert lists to numerical arrays
angles_deg = np.asarray(angles_deg, dtype=float)
values_mean = np.asarray(values_mean, dtype=float)
values_std = np.asarray(values_std, dtype=float)

angles = np.deg2rad(angles_deg)

# Mean ± standard-deviation boundaries
lower = np.maximum(values_mean - values_std, 0)
upper = values_mean + values_std

# Close the polar curves
angles_closed = np.append(angles, angles[0])
mean_closed = np.append(values_mean, values_mean[0])
lower_closed = np.append(lower, lower[0])
upper_closed = np.append(upper, upper[0])

# Good size for a two-column-wide figure
fig, ax = plt.subplots(
    figsize=(6.8, 6.2),
    subplot_kw={"projection": "polar"},
    layout="constrained"
)

# Keep the physical rotation convention
ax.set_theta_zero_location("N")  # 0° at top
ax.set_theta_direction(-1)       # Increasing angles clockwise

# Mean uncertainty band
ax.fill_between(
    angles_closed,
    lower_closed,
    upper_closed,
    color="#6A3D9A",
    alpha=0.20,
    linewidth=0,
    label="Mean ± SD"
)

# Mean line
ax.plot(
    angles_closed,
    mean_closed,
    color="#6A3D9A",
    linewidth=2.0,
    marker="o",
    markerfacecolor="white",
    markeredgewidth=1.2,
    label="Mean MAE"
)

# Angular labels
ax.set_thetagrids(
    angles_deg,
    labels=[f"{int(a)}°" for a in angles_deg],
    fontsize=8
)

# Radial scale: set this from your actual MAE range
ax.set_rticks([5, 10, 15, 20, 25])
ax.set_rlabel_position(22.5)
ax.tick_params(axis="y", labelsize=8)

# Subtle grid, suitable for print
ax.grid(color="0.75", linewidth=0.6, alpha=0.8)

# Usually omit a long title in the submitted figure:
# put description in the manuscript figure caption instead.
# ax.set_title("MAE by camera angle", pad=18)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=2,
    frameon=False,
    handlelength=2.2,
    columnspacing=1.4
)

# Vector format: best for papers
fig.savefig(
    os.path.join(save_dir, f"MAE_{level}_{joint_name}.pdf"),
    bbox_inches="tight"
)

# High-resolution raster version: useful if the journal requests PNG/TIFF
fig.savefig(
    os.path.join(save_dir, f"MAE_{level}_{joint_name}.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)