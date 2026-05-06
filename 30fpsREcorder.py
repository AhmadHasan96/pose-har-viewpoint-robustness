import cv2

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("logitune_video.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(cap.get(3))
print(cap.get(4))

width = int(cap.get(3))
height = int(cap.get(4))

print(width)
print(height)

# window_name = "Pose Viewer"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 1200, 700)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = f"New_vidoes/No_mediaPipe_video.avi"
out = cv2.VideoWriter(video_path, fourcc, 90, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()