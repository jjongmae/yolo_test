from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "D:\data\여주시험도로_20250610\카메라1_202506101340.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

frame_count = 0  # 프레임 인덱스

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        print("Failed to read frame or end of video reached.")
        break

    frame_count += 1

    if frame_count % 4 != 0:
        continue
    
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    result = model.track(frame, persist=True)[0]

    # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Visualize the result on the frame
        frame = result.plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()