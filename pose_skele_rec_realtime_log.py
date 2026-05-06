import cv2
import mediapipe as mp
import time
import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # BRIO Webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # PC Webcam


# Force resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(cap.get(3))
print(cap.get(4))

window_name = "Pose Viewer"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 700)

recording = False
waiting = False
out = None
csv_file = None
csv_writer = None
frame_id = 0
start_time = None
recording_start_time = None
wait_start_time = None
total = 0
count = 0
all_lengths = []
record_duration = 20
delay_before_start = 10

bones = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (23, 24)             # hips
]
bone_names = [
    "L_sh_el", "L_el_wr",
    "R_sh_el", "R_el_wr",
    "shoulders",
    "L_hp_kn", "L_kn_an",
    "R_hp_kn", "R_kn_an",
    "hips"
]
def get_bone_lengths(landmarks):
    lengths = []
    for i, j in bones:
        p1 = np.array([landmarks[i].x, landmarks[i].y])
        p2 = np.array([landmarks[j].x, landmarks[j].y])
        lengths.append(np.linalg.norm(p1 - p2))
    return np.array(lengths)

os.makedirs("New_vidoes", exist_ok=True)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, 
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Non-blocking delay
        if waiting:
            if time.time() - wait_start_time >= delay_before_start:
                print("🔴 Recording started")

                now = datetime.now()
                date_str = now.strftime("%d-%m-%Y")
                time_str = now.strftime("%H-%M-%S")
                filename_base = f"{date_str}_{time_str}"

                video_path = f"New_vidoes/{filename_base}.avi"
                csv_path = f"New_vidoes/{filename_base}.csv"

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                fps = 15

                width = int(cap.get(3))
                height = int(cap.get(4))
                print(width)
                print(height)
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) # comment if no video recording

                csv_file = open(csv_path, mode='w', newline='')
                csv_writer = csv.writer(csv_file)

                header = ["date", "time", "rel_time_sec", "frame"]
                for i in range(33):
                    header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
                csv_writer.writerow(header)

                frame_id = 0
                all_lengths = []
                start_time = time.time()
                recording_start_time = time.time()

                recording = True
                waiting = False

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb) # comment to reomve MediaPipe
        frame_display = frame.copy() # to display frames no skele in realtime
        frame_record = frame.copy() # to record frames with skele

        if results.pose_landmarks: # comment all the if to remove MEdiaPipe
            mp_drawing.draw_landmarks(
                frame_record,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        # ✅ Video writing
        if recording and out is not None:
            out.write(frame_record)

        # ✅ CSV logging
        if recording and results.pose_landmarks and csv_writer is not None: # comment to remove MEdiaPIpe
        # if recording and csv_writer is not None:

            now = datetime.now()
            date_val = now.strftime("%d-%m-%Y")
            time_val = now.strftime("%H:%M:%S")

            rel_time_sec = time.time() - start_time

            lengths = get_bone_lengths(results.pose_landmarks.landmark)
            all_lengths.append(lengths)

            row = [date_val, time_val, rel_time_sec, frame_id]

            for lm in results.pose_landmarks.landmark:  #comment to remove MediaPipe
            # for lm in results.pose_world_landmarks.landmark:
                row += [lm.x, lm.y, lm.z, lm.visibility] #comment to remove MEdiaPipe
                
                total += lm.visibility
                count += 1
            
            csv_writer.writerow(row)
            frame_id += 1

        # ✅ Auto stop
        if recording and recording_start_time is not None:
            if time.time() - recording_start_time >= record_duration:
                recording = False
                fps_real = frame_id / (time.time() - start_time)

                avg_visibility = total / count

                all_lengths = np.array(all_lengths)
                L_ref = np.mean(all_lengths, axis=0)
                errors = np.abs(all_lengths - L_ref) / L_ref
                blc_per_frame = np.mean(errors, axis=1)
                blc = np.mean(errors)

                
                print(f"{len(all_lengths)} = {frame_id}")
                print(f"Bone Length Error = {blc}")
                print(f"avg v {avg_visibility} = total: {total} / count: {count}")
                print(f"real_fps: {fps_real}")
                print("⏹ Recording stopped")

                # for i in range(all_lengths.shape[1]):
                #     plt.plot(all_lengths[:, i], label=bone_names[i])
                plt.plot(blc_per_frame)
                plt.ylim(0, 1)
                # plt.xlabel("Frame")
                # plt.ylabel("Length")
                # plt.title("Bone Length Consistency over Time")

                plt.savefig(f"New_vidoes/{filename_base}.png", dpi=300, bbox_inches='tight')
                plt.close()

                if out:
                    out.release()
                    out = None

                if csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None

        status = "REC ON" if recording else ("WAIT..." if waiting else "REC OFF")
        # cv2.putText(frame_display, status, (20, 40), # showing REC ON & REC OFF on display no skele
        cv2.putText(frame_record, status, (20, 40), # showing REC ON & REC OFF on display with skele
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if recording else (0, 255, 255) if waiting else (200, 200, 200), 2)

        # cv2.imshow(window_name, frame_display) # don't show skele on screen realtime!
        cv2.imshow(window_name, frame_record) # show skele on screen realtime! 

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            if not recording and not waiting:
                print("⏳ Waiting 5 seconds...")
                waiting = True
                wait_start_time = time.time()

            elif recording:
                recording = False
                fps_real = frame_id / (time.time() - start_time)
                avg_visibility = total / count
                print(f"avg v {avg_visibility} = total: {total} / count: {count}")
                print(f"real_fps: {fps_real}")
                print("⏹ Recording stopped (manual)")

                if out:
                    out.release()
                    out = None

                if csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None

cap.release()
if out:
    out.release()
if csv_file:
    csv_file.close()
cv2.destroyAllWindows()