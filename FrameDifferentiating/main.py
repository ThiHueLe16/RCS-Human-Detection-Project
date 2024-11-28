import cv2
from FrameDifferentiating import FrameDifferencing
from FrameDifferentiating.FrameDifferencing import FrameDifferencing

# Open video capture (replace with your video file or camera feed)
cap = cv2.VideoCapture("../testYolo/peopleTestYolo.mp4")

# Initialize the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Failed to read video")
    cap.release()
    exit()

# Convert the first frame to grayscale
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    # Capture the next frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the second frame to grayscale
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Get object movement detections
    detections = FrameDifferencing.get_detections(frame1_gray, frame2_gray)

    # Draw bounding boxes around the detected moving objects
    FrameDifferencing.draw_bboxes(frame2, detections)

    # Display the result
    cv2.imshow("Motion Detection", frame2)

    # Update frame1 for the next iteration
    frame1_gray = frame2_gray

    # Exit on pressing ESC
    if cv2.waitKey(1) == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
