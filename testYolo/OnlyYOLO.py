import cv2
from ultralytics import YOLO

# Initialize the YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8m.pt")  # Use YOLOv8 nano model (or any other you want)

# Object class mapping for YOLOv8 COCO dataset
# Class 0 corresponds to "person" in the COCO dataset
names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
         8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
         14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
         22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
         29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
         35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
         48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
         55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
         62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
         69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
         76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture("./test2")  # Use 0 for webcam or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the current frame
    results = model(frame)

    # Iterate over the detections
    for detection in results[0].boxes:
        # Extract class ID and bounding box coordinates
        class_id = int(detection.cls)
        bbox = detection.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

        # Check if the detected object is "person" (class_id == 0)
        if class_id == 0:
            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, names[class_id], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Person Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
