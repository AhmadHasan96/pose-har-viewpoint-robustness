import cv2
import time
import os
from datetime import datetime

CAMERA_INDEX = 0
USE_DSHOW = True
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_FPS = 30
RECORD_DURATION = 20
DELAY_BEFORE_START = 10
OUTPUT_DIR = "New_videos"
CODEC = "MJPG"  # try MJPG first for higher fps; fallback to XVID if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

backend = cv2.CAP_DSHOW if USE_DSHOW else 0
cap = cv2.VideoCapture(CAMERA_INDEX, backend)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
reported_fps = cap.get(cv2.CAP_PROP_FPS)

window_name = "Video Recorder"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 700)

recording = False
waiting = False
out = None
frame_id = 0
start_time = None
recording_start_time = None
wait_start_time = None
filename_base = None
video_path = None


def start_recording():
    global recording, waiting, out, frame_id, start_time, recording_start_time, filename_base, video_path

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H-%M-%S")
    filename_base = f"{date_str}_{time_str}"
    video_path = f"{OUTPUT_DIR}/{filename_base}.avi"

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (actual_width, actual_height))

    frame_id = 0
    start_time = time.time()
    recording_start_time = time.time()
    recording = True
    waiting = False

    print(f"🔴 Recording started: {video_path}")
    print(f"Requested FPS: {TARGET_FPS} | Camera reported FPS: {reported_fps}")
    print(f"Resolution: {actual_width}x{actual_height}")


def stop_recording(manual=False):
    global recording, out

    if not recording:
        return

    recording = False
    elapsed = max(time.time() - start_time, 1e-9)
    fps_real = frame_id / elapsed
    label = "manual" if manual else "auto"

    print(f"⏹ Recording stopped ({label})")
    print(f"Saved frames: {frame_id}")
    print(f"Measured write FPS: {fps_real:.2f}")

    if out:
        out.release()
        out = None


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    if waiting and time.time() - wait_start_time >= DELAY_BEFORE_START:
        start_recording()

    frame_display = frame.copy()

    if recording and out is not None:
        out.write(frame)
        frame_id += 1

    if recording and recording_start_time is not None:
        if time.time() - recording_start_time >= RECORD_DURATION:
            stop_recording(manual=False)

    status = "REC ON" if recording else ("WAIT..." if waiting else "REC OFF")
    color = (0, 0, 255) if recording else (0, 255, 255) if waiting else (200, 200, 200)

    cv2.putText(frame_display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame_display, f"Target FPS: {TARGET_FPS}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_display, f"Camera FPS: {reported_fps:.2f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_display, f"Resolution: {actual_width}x{actual_height}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(window_name, frame_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        if not recording and not waiting:
            print(f"⏳ Waiting {DELAY_BEFORE_START} seconds...")
            waiting = True
            wait_start_time = time.time()
        elif recording:
            stop_recording(manual=True)

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()