import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLO model
model = YOLO("yolov8m.pt")

# Initialize DeepSort tracker
deepsort = DeepSort(max_age=30, nn_budget=70, max_iou_distance=0.7)

# Video capture
cap = cv2.VideoCapture("../testYolo/peopleTestYolo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 inference
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

    # Prepare detections for DeepSort
    detections_input = [
        [x1, y1, x2, y2, conf, int(cls)]
        for (x1, y1, x2, y2), conf, cls in zip(detections, confidences, class_ids)
    ]

    # Update DeepSort
    tracks = deepsort.update_tracks(detections_input, frame)

    # Draw results
    for track in tracks:
        bbox = track.to_tlbr()  # Convert to `[x1, y1, x2, y2]`
        track_id = track.track_id

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()


