import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

level = "BirdView"
joint_name = "R_shoulder_deg"
model_complex = 1
# save_dir = f"New_videos/August/{level} Comparision Output_{model_complex}"
save_dir = f"New_videos/July/14.07.2026/{level} Comparision Output_{model_complex}"
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

##################################### Feet Level ##########################################
# values_mean = [17.7, 21.64, 6.18, 8.61, 16.13, 14.29, 5.98, 10.38] #  L_shoulder_deg
# values_std = [23.49, 9.27, 2.15, 0.72, 5.11 , 6.11 , 2.98, 2.24]

# values_mean = [14.32, 18.56, 1.5, 9.1, 11.8, 5.77, 2.08, 9.58] #  R_shoulder_deg
# values_std = [24.05, 8.85, 1.98, 2.04, 3.93 , 7.86 , 2.16, 1.86]

# values_mean = [4.86, 9.42, 11.76, 4.37, 12.01, 6.19, 1.98, 7.39] #  L_hip_deg
# values_std = [8.1, 0.96, 0.25, 1.36, 1.02 , 1.23 , 0.57, 0.59]

# values_mean = [3.61, 2.03, 1.94, 8.7, 11.06, 0.73, 5.69, 2.34] #  R_hip_deg
# values_std = [4.63, 2.21, 0.42, 0.64, 0.99 , 1.03 , 0.75, 0.95]

##################################### Hip Level ##########################################
# values_mean = [11.05, 10.07, 9.64, 5.05, 13.12, 5.98, 14.62, 13.39] #  L_shoulder_deg
# values_std = [5.22 ,4.81 , 1.73, 1.12 , 2.26, 2.19, 3.23, 1.97]

# values_mean = [7.31, 14.87, 9.48, 4.18, 9.29, 6.37, 7.07,9.25] #  R_shoulder_deg
# values_std = [6.6, 5.47, 2.93, 1.63, 3.93, 1.56, 1.86, 1.99]

# values_mean = [8.35, 4.71, 9.36, 8.01, 3.57, 2.37,2.71, 9.07] #  L_hip_deg
# values_std = [0.43, 0.96, 0.69, 0.36, 0.6, 0.45, 0.81, 0.72]

# values_mean = [6.33, 2.3, 2.58, 2.77, 5.39, 3.97, 7.9, 2.41] #  R_hip_deg
# values_std = [0.63, 1.01, 1.18, 0.33, 0.63, 0.48, 0.7, 0.55]


##################################### Chest Level Faulty ##########################################
# values_mean = [7.14, 4.06, 10.07, 4.72, 15.14, 6.82, 10.21, 11.2] #  L_shoulder_deg
# values_std = [2.13, 4.22, 1.03, 0.97 , 5.45, 4.16, 5.06, 2.46]

# values_mean = [7.22, 7.62, 10.6, 6.79, 15, 13.6, 8.14, 5.34] #  L_shoulder_deg_2
# values_std = [2.48, 1.23, 1.25, 1.63, 3.48, 2.76, 1.82, 2.48]

# values_mean = [7.27, 9.82, 22.03, 7.13, 9.88, 6.77, 6.3, 7.71] #  R_shoulder_deg
# values_std = [3.37, 4.91, 3.57, 1.19, 4.81, 2.76, 2.79, 2.81]

# values_mean = [21.2, 19.3, 17.5, 4.53, 8.98, 8.79, 10.84, 15.12] #  R_shoulder_deg_2
# values_std = [2.7, 0.98, 3.71, 1.21, 2.74, 1.71, 0.84, 1.84]

# values_mean = [8.74, 4.44, 4.24, 7.68, 4.12, 2.6, 1.66, 5.72] #  L_hip_deg
# values_std = [0.8, 0.88, 1.977, 0.38, 0.93, 0.68, 0.77, 0.45]

# values_mean = [5, 3.59, 6.08, 8.5, 2.88, 4.37, 13.96, 10.22] #  L_hip_deg_2
# values_std = [0.46, 0.33, 3.2, 0.19, 0.43, 0.38, 0.28, 0.39]

# values_mean = [4.92, 2.73, 42.64, 4.28, 4.92, 6.48, 5.58,2.35] #  R_hip_deg
# values_std = [1.03, 0.86, 19.69,0.52, 1.64, 1.25, 1.33, 0.84]

# values_mean = [4.92, 10.24, 55.78, 17.93, 14.51, 5.12, 10.09, 7.84] #  R_hip_deg_2
# values_std = [0.69, 0.73, 20.7, 0.82, 1.07, 1.31, 0.44, 0.59]

##################################### Chest Level Repeated ##########################################
# values_mean = [7.1, 5.96, 5.58, 4.5, 9.46, 12.68, 8.25, 8.32] #  L_shoulder_deg_0
# values_std = [1.98, 2.98, 1.88, 1.67, 3.1, 2.24, 2.75, 1.44]

# values_mean = [6.16, 4.11, 6.76, 5.37, 15.84, 10.41, 12.53, 11.11] #  L_shoulder_deg
# values_std = [1.59, 2.94, 1.24, 1.17, 2.84, 2.49, 3.13,1.22]

# values_mean = [9.64, 9.44, 11.5, 9.3, 13.89, 12.29, 10.99, 4.71] #  L_shoulder_deg_2
# values_std = [1.64, 1.84, 1.96, 1.39, 3.46, 2.86, 1.85, 0.96]

# values_mean = [5.5, 13.2, 12.49, 15.01, 9.04, 6.61, 7.05, 7.88] #  R_shoulder_deg_0
# values_std = [2.38, 3.82, 2.03, 1.89, 3.47, 1.98, 2.28, 1.73]

# values_mean = [5.64, 10.44, 9.9, 4.45, 11.16, 6.51, 7.4, 7.52] #  R_shoulder_deg
# values_std = [1.88, 3.43, 1.85, 1.64, 3.42, 1.81, 1.93, 1.55]

# values_mean = [16.64, 18.66, 18.68, 5.61, 7.14, 7.97, 11.78, 12.57] #  R_shoulder_deg_2
# values_std = [1.14, 1.21, 2.89, 0.82, 2.73, 1.96, 1.32, 0.78]

# values_mean = [5.71, 5.49, 10.5, 3.18, 2.52, 3.32, 5.02, 11.82] #  L_Hip_deg_0
# values_std = [0.49, 0.55, 0.42, 0.2, 0.29, 0.61, 0.2, 0.69]

# values_mean = [9, 5.5, 11.46, 8.23, 3.12, 2.23, 1.72, 8.55] #  L_Hip_deg
# values_std = [0.51, 0.89, 0.633, 0.38, 0.54, 0.65, 0.49, 0.46]

# values_mean = [4.78, 4.02, 5.69, 7.66, 6.03, 2.58, 12.72, 10.09] #  L_Hip_deg_2
# values_std = [0.61, 0.5, 0.44, 0.48, 0.33, 0.91, 0.38, 0.34]

# values_mean = [7.62, 2.81, 3.43, 16.08, 3.05, 2.06, 4.79, 3.33] #  R_hip_deg_0
# values_std = [1.8, 0.99, 0.66, 0.55, 0.59, 0.67, 0.34, 0.74]

# values_mean = [8.5, 2.1, 2.08, 3.01, 2.92, 7.43, 5.14, 3.62] #  R_hip_deg
# values_std = [1.47, 0.87, 0.89, 0.34, 0.71, 0.68, 0.69, 1.17]

# values_mean = [8.01, 11.11, 13.14, 21.19, 5.78, 2.72, 12.59, 8.53] #  R_hip_deg_2
# values_std = [1.53, 1.03, 0.72, 0.63, 0.48, 0.97, 0.55, 0.38]

##################################### Chest Level Uniform Background ##########################################
# values_mean = [10.35, 5.27, 7.76, 7.87, 14.51, 7.62, 14.84, 12.25] #  L_shoulder_deg
# values_std = [3.74, 4.54, 4.64, 5.34, 4.43, 7.69, 8.03, 8.59]

# values_mean = [10.8, 9.31, 10.48, 7.57, 9.14, 4.45, 7.61, 6.48 #  R_shoulder_deg
# values_std = [5.48, 4.39, 6.39, 4.41, 4.8, 6.1, 5.82, 7.12]

# values_mean = [10.07, 5.04, 10.14] #  L_Hip_deg
# values_std = [1.02, 0.44, 0.61]

# values_mean = [4.81, 2.35, 4.58] #  R_hip_deg
# values_std = [0.78, 1.12, 1.13]

##################################### Head Level ##########################################
# values_mean = [8.18, 2.87, 10.69, 7.7, 17.1, 11.35, 12.67, 9.46] #  L_shoulder_deg
# values_std = [ 2.87, 1.01 ,1.45 , 1.52, 5.02, 4.12, 2.52 , 1.66]

# values_mean = [6.05, 9.73, 14.07, 11.25, 14.31, 16.7, 8.93, 5.5] #  L_shoulder_deg_2
# values_std = [1.77, 1.46, 1.08, 1.87, 3.92, 2.79, 2.07, 1.13]

# values_mean = [3.66, 7.66, 9.28, 5.38, 9, 4.17, 6.13, 7.88] #  R_shoulder_deg
# values_std = [3.19, 0.84, 2.43, 1.69, 4.77, 0.79, 1.22, 1.89]

# values_mean = [13.61, 18.45, 13.21, 3.47, 4.59, 8.97, 9.63, 13.87] #  R_shoulder_deg_2
# values_std = [0.91, 0.88, 1.12, 1.27, 2.4, 2.01, 1.01, 0.96]

# values_mean = [8.13, 5.01, 9.88, 7.4, 2.7, 2.33, 3.2, 5.65] #  L_hip_deg
# values_std = [0.88, 0.53, 1.07, 0.27, 1.14, 0.65, 0.72, 0.61]

# values_mean = [] #  L_hip_deg_2
# values_std = []

# values_mean = [4.09, 3.42, 5.38, 0.91, 4.17, 3.12, 1.47, 2.31] #  R_hip_deg
# values_std = [0.97, 0.44, 4.89, 0.6, 0.799, 0.355, 1.24, 1.09]

# values_mean = [] #  R_hip_deg_2
# values_std = []

##################################### Bird Level ##########################################
# values_mean = [10.08, 8.44, 12.35, 12.69, 26.2, 20.44, 10.64, 9.26] #  L_shoulder_deg
# values_std = [ 1.46,1.46 , 1.89, 4.12, 3.18, 1.81,4.61 ,1.32 ]

values_mean = [8.05, 8.04, 12.79, 5.99, 16.33, 10.08, 7.64, 13.32] #  R_shoulder_deg
values_std = [1.62, 1.84, 2.47, 4.09, 3.28, 1.38, 3.13, 1.55]

# values_mean = [5.73, 4.44, 13.75, 5.35, 1.85, 3.93, 2.95, 4.85] #  L_Hip_deg
# values_std = [2.11, 1.12, 1.13, 0.59, 0.69, 0.78, 1.61, 1.08]

# values_mean = [4.43, 4.33, 2.08, 8.34, 5.72, 1.03, 9.11, 6.18] #  R_hip_deg
# values_std = [1.38, 1.09, 1.15, 1.16, 1.22, 0.71, 1.31, 0.47]


angles = np.deg2rad(angles_deg)

values_mean = np.asarray(values_mean, dtype=float)
values_std = np.asarray(values_std, dtype=float)

angles_closed = np.append(angles, angles[0])
values_closed = np.append(values_mean, values_mean[0])

lower = np.maximum(values_mean - values_std, 0)
upper = values_mean + values_std

lower_closed = np.append(lower, lower[0])
upper_closed = np.append(upper, upper[0])

fig, ax = plt.subplots(
    figsize=(7, 7),
    subplot_kw={"projection": "polar"}
)

ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")

ax.plot(
    angles_closed,
    values_closed,
    color="purple",
    linewidth=2.5,
    marker="o",
    label="Mean MAE"
)

ax.fill_between(
    angles_closed,
    lower_closed,
    upper_closed,
    color="purple",
    alpha=0.20,
    label="Mean ± STD"
)


# Radial numbers: e.g., 5, 10, 15, 20, 25
ax.set_rticks([5, 10, 15, 20, 25])
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.set_major_formatter('{x:g}°')
ax.set_rlabel_position(70) 

ax.tick_params(axis="x", labelsize=11)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")
# ax.xaxis.set_major_formatter('[{x:g}]')

ax.set_title(
    f"MAE [{level}] by angle [°]:\n\n"
    f"Mean = {np.mean(values_mean):.2f}° | "
    f"STD = {np.mean(values_std):.2f}°",
    fontsize=13,
    fontweight="bold",
    pad=20
)

ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),fontsize=13,
    frameon=False)

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, f"MAE {level} {joint_name} MC {model_complex}.svg"),
    dpi=300,
    bbox_inches="tight"
)

summary_overall = pd.DataFrame({
    "Angle [°]": angles_deg,
    "MAE [°]": values_mean,
    "STD [°]" : values_std,
})

summary_overall.to_csv(os.path.join(save_dir, f"{level}_{joint_name}_MC{model_complex}.csv"), encoding='utf-8-sig', index=False)