import cv2
import numpy as np
from ultralytics import YOLO


def detect_motion_with_optical_flow(frame1, frame2):
    """
    Detects motion between two grayscale frames using optical flow.
    """
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # Compute magnitude and direction
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold motion magnitude to create a motion mask
    motion_mask = (magnitude > 3.0).astype(np.uint8) * 255  # Adjust threshold if needed, hihger magnitude means ignore smaller motion change
    return motion_mask


# Initialize video capture
video_path = "../testYolo/test2.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load the YOLO model
model = YOLO("yolov8m.pt")  # Load the pre-trained YOLOv8 model

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frames are left

    # Convert the frame to grayscale for optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect motion between the previous and current frame
    motion_mask = detect_motion_with_optical_flow(prev_gray, gray)

    # Use YOLOv8 to detect objects in the frame
    results = model(frame)
    result = results[0]
    bounding_boxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    # Draw green bounding boxes for YOLO detections
    for class_, box in zip(classes, bounding_boxes):
        (x, y, x2, y2) = box
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)  # Green for detected object
        cv2.putText(frame, f'{model.names[class_]}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Find contours in the motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each detected motion, check if it overlaps with any YOLO detections
    for contour in contours:
        # Get bounding box for motion contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the motion bounding box overlaps with any YOLO object box
        overlap = False
        for (bx, by, bx2, by2) in bounding_boxes:
            # Check if the motion is within the bounding box of a detected object
            if not (x + w < bx or x > bx2 or y + h < by or y > by2):  # Check for overlap
                overlap = True
                break

        # If no overlap, draw a red bounding box for the motion
        if not overlap:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for detected motion

    # Display the result frame
    cv2.imshow("Detected Objects and Motion", frame)

    # Update the previous frame for optical flow detection
    prev_gray = gray

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
