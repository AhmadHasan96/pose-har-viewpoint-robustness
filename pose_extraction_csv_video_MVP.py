import cv2
import mediapipe as mp
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from mpl_toolkits.mplot3d import Axes3D

# =========================
# SETTINGS
# =========================

VIDEO_PATH = "videos/lifting_front_normal.mp4"

CSV_DIR = "Unified_csv"
RENDER_DIR = "Unified_rendered"
ANIMATION_DIR = "Unified_ani"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(ANIMATION_DIR, exist_ok=True)

# =========================
# FLAGS
# =========================

GENERATE_CSV = False #True
GENERATE_VIDEO = False #True
PLOT_3D = True

PLOT_FRAME = 0

# =========================
# FILE PATHS
# =========================

video_name = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]

csv_path = os.path.join(
    CSV_DIR,
    f"{video_name}_world_landmarks.csv"
)

render_path = os.path.join(
    RENDER_DIR,
    f"{video_name}_skeleton.mp4"
)

# =========================
# EXISTENCE CHECKS
# =========================

csv_exists = os.path.exists(csv_path)
video_exists = os.path.exists(render_path)

if csv_exists:
    print("CSV already exists")
else:
    print("CSV does not exist")

if video_exists:
    print("Skeleton video already exists")
else:
    print("Skeleton video does not exist")

# Skip generation if file already exists
GENERATE_CSV = GENERATE_CSV and not csv_exists
GENERATE_VIDEO = GENERATE_VIDEO and not video_exists

# =========================
# INIT MEDIAPIPE
# =========================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# LOAD EXISTING CSV
# =========================

if not GENERATE_CSV and csv_exists:

    print("Loading existing CSV...")

    df = pd.read_csv(csv_path)

# =========================
# PROCESS VIDEO
# =========================

else:

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise Exception(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # =====================
    # VIDEO WRITER
    # =====================

    if GENERATE_VIDEO:

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            render_path,
            fourcc,
            fps,
            (width, height)
        )

    landmarks_data = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = pose.process(frame_rgb)

        # =================
        # CSV EXTRACTION
        # =================

        frame_time_s = frame_count / fps

        frame_landmarks = [
            frame_count,
            frame_time_s
        ]

        if result.pose_world_landmarks:

            for lm in result.pose_world_landmarks.landmark:

                frame_landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ])

        else:

            for _ in range(33):

                frame_landmarks.extend([
                    None,
                    None,
                    None,
                    None
                ])

        landmarks_data.append(frame_landmarks)

        # =================
        # VIDEO RENDERING
        # =================

        if GENERATE_VIDEO:

            draw_frame = frame.copy()

            if result.pose_landmarks:

                mp_draw.draw_landmarks(
                    draw_frame,
                    result.pose_landmarks,
                    POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(100, 255, 0),
                        thickness=2,
                        circle_radius=1
                    ),
                    mp_draw.DrawingSpec(
                        color=(0, 100, 255),
                        thickness=2,
                        circle_radius=4
                    )
                )

            out.write(draw_frame)

        frame_count += 1

    # =====================
    # CLEANUP
    # =====================

    cap.release()

    if GENERATE_VIDEO:
        out.release()

    pose.close()

    # =====================
    # SAVE CSV
    # =====================

    columns = ["frame", "time_s"]

    for i in range(33):

        columns.extend([
            f"x_{i}",
            f"y_{i}",
            f"z_{i}",
            f"v_{i}"
        ])

    df = pd.DataFrame(
        landmarks_data,
        columns=columns
    )

    if GENERATE_CSV:

        df.to_csv(
            csv_path,
            index=False
        )

        print(f"Saved CSV -> {csv_path}")

    if GENERATE_VIDEO:

        print(
            f"Saved skeleton video -> {render_path}"
        )

# =========================
# 3D PLOTTING
# =========================

def plot_3d_skeleton(df, frame_idx):

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    row = df.iloc[frame_idx]

    xs = []
    ys = []
    zs = []

    for i in range(33):

        xs.append(row[f"x_{i}"])

        # Flip Y for natural orientation
        ys.append(row[f"z_{i}"])

        zs.append(-row[f"y_{i}"])

        # ys.append(-row[f"y_{i}"])
        # zs.append(row[f"z_{i}"])

    # =====================
    # JOINTS
    # =====================

    ax.scatter(xs, ys, zs, s=50)

    # =====================
    # BONES
    # =====================

    for start, end in POSE_CONNECTIONS:

        if (
            pd.notna(xs[start])
            and
            pd.notna(xs[end])
        ):

            ax.plot(
                [xs[start], xs[end]],
                [ys[start], ys[end]],
                [zs[start], zs[end]]
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"3D Skeleton - Frame {frame_idx}"
    )

    plt.show()

# =========================
# SHOW 3D PLOT
# =========================

if PLOT_3D:

    if PLOT_FRAME >= len(df):

        print(
            f"Frame {PLOT_FRAME} does not exist"
        )

    else:

        plot_3d_skeleton(
            df,
            PLOT_FRAME
        )
def animate_3d_skeleton_old(
    df,
    fps=30,
    save_animation=True
):

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    # =====================
    # UPDATE FUNCTION
    # =====================

    def update(frame_idx):

        ax.cla()

        row = df.iloc[frame_idx]

        xs = []
        ys = []
        zs = []

        for i in range(33):

            # =================
            # MEDIAPIPE AXES
            # =================

            mp_x = row[f"x_{i}"]
            mp_y = row[f"y_{i}"]
            mp_z = row[f"z_{i}"]

            # =================
            # REMAP FOR HUMAN VIEW
            # =================

            plot_x = mp_x

            # depth
            plot_y = -mp_z

            # height
            plot_z = -mp_y

            xs.append(plot_x)
            ys.append(plot_y)
            zs.append(plot_z)

        # =====================
        # DRAW JOINTS
        # =====================

        ax.scatter(
            xs,
            ys,
            zs,
            s=40
        )

        # =====================
        # DRAW BONES
        # =====================

        for start, end in POSE_CONNECTIONS:

            if (
                pd.notna(xs[start])
                and
                pd.notna(xs[end])
            ):

                ax.plot(
                    [xs[start], xs[end]],
                    [ys[start], ys[end]],
                    [zs[start], zs[end]]
                )

        # =====================
        # AXES
        # =====================

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax.set_xlabel("Left / Right")
        ax.set_ylabel("Depth")
        ax.set_zlabel("Height")

        ax.set_title(
            f"3D Skeleton Animation - Frame {frame_idx}"
        )

        # Better proportions
        ax.set_box_aspect([1, 1, 2])

        # Better camera angle
        ax.view_init(
            elev=15,
            azim=-70
        )

    # =====================
    # CREATE ANIMATION
    # =====================

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000 / fps,
        repeat=True
    )

    # =====================
    # SAVE
    # =====================

    if save_animation:

        output_path = os.path.join(
            ANIMATION_DIR,
            f"{video_name}_3d.gif"
        )

        anim.save(
            output_path,
            writer="pillow",
            fps=fps
        )

        print(
            f"Saved animation -> {output_path}"
        )

    plt.show()


def animate_3d_skeleton(
    df,
    fps=30,
    save_animation=True
):

    fig = plt.figure(figsize=(14, 10))

    # =========================
    # SUBPLOTS
    # =========================

    # ax3d = fig.add_subplot(
    #     221,
    #     projection='3d'
    # )
    #
    # ax_front = fig.add_subplot(222)
    #
    # ax_side = fig.add_subplot(212)
    #

    # =========================
    # LAYOUT
    # =========================

    gs = GridSpec(
        2,
        2,
        width_ratios=[1.5, 1],
        height_ratios=[1, 1]
    )

    # Large 3D pane on left
    ax3d = fig.add_subplot(
        gs[:, 0],
        projection='3d'
    )

    # Front view top-right
    ax_front = fig.add_subplot(
        gs[0, 1]
    )

    # Side view bottom-right
    ax_side = fig.add_subplot(
        gs[1, 1]
    )

    # =========================
    # UPDATE FUNCTION
    # =========================

    def update(frame_idx):

        ax3d.cla()
        ax_front.cla()
        ax_side.cla()

        row = df.iloc[frame_idx]

        xs = []
        ys = []
        zs = []

        # =====================
        # LOAD LANDMARKS
        # =====================

        for i in range(33):

            mp_x = row[f"x_{i}"]
            mp_y = row[f"y_{i}"]
            mp_z = row[f"z_{i}"]

            # =================
            # REMAP AXES
            # =================

            x = mp_x

            # depth
            y = -mp_z

            # height
            z = -mp_y

            xs.append(x)
            ys.append(y)
            zs.append(z)

        # ==================================================
        # 3D VIEW
        # ==================================================

        ax3d.scatter(xs, ys, zs, s=40)

        for start, end in POSE_CONNECTIONS:

            if (
                pd.notna(xs[start])
                and
                pd.notna(xs[end])
            ):

                ax3d.plot(
                    [xs[start], xs[end]],
                    [ys[start], ys[end]],
                    [zs[start], zs[end]]
                )

        ax3d.set_xlim(-1, 1)
        ax3d.set_ylim(-1, 1)
        ax3d.set_zlim(-1, 1)

        ax3d.set_xlabel("Left/Right")
        ax3d.set_ylabel("Depth")
        ax3d.set_zlabel("Height")

        ax3d.set_title("3D View")

        ax3d.set_box_aspect([1, 1, 2])

        ax3d.view_init(
            elev=15,
            azim=-70
        )

        # ==================================================
        # FRONT VIEW
        # X vs HEIGHT
        # ==================================================

        ax_front.scatter(xs, zs)

        for start, end in POSE_CONNECTIONS:

            if (
                pd.notna(xs[start])
                and
                pd.notna(xs[end])
            ):

                ax_front.plot(
                    [xs[start], xs[end]],
                    [zs[start], zs[end]]
                )

        ax_front.set_xlim(-1, 1)
        ax_front.set_ylim(-1, 1)

        ax_front.set_xlabel("Left/Right")
        ax_front.set_ylabel("Height")

        ax_front.set_title("Front View")

        ax_front.set_aspect('equal')
        ax_front.grid(True)

        # ==================================================
        # SIDE VIEW
        # DEPTH vs HEIGHT
        # ==================================================

        ax_side.scatter(ys, zs)

        for start, end in POSE_CONNECTIONS:

            if (
                pd.notna(ys[start])
                and
                pd.notna(ys[end])
            ):

                ax_side.plot(
                    [ys[start], ys[end]],
                    [zs[start], zs[end]]
                )

        ax_side.set_xlim(-1, 1)
        ax_side.set_ylim(-1, 1)

        ax_side.set_xlabel("Depth")
        ax_side.set_ylabel("Height")

        ax_side.set_title("Side View")

        ax_side.set_aspect('equal')
        ax_side.grid(True)

        # ==================================================
        # GLOBAL TITLE
        # ==================================================

        fig.suptitle(
            f"Pose Animation - Frame {frame_idx}",
            fontsize=16
        )

    # =========================
    # CREATE ANIMATION
    # =========================

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000 / fps,
        repeat=True
    )

    # =========================
    # SAVE
    # =========================

    if save_animation:

        output_path = os.path.join(
            ANIMATION_DIR,
            f"{video_name}_multiview.gif"
        )

        anim.save(
            output_path,
            writer="pillow",
            fps=fps
        )

        print(
            f"Saved animation -> {output_path}"
        )

    plt.show()


animate_3d_skeleton(
    df,
    fps=30, #fps,
    #fps=fps,
    save_animation=True
)
# define the fps for the animation