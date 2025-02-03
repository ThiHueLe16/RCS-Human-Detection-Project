# This file is only for testing the function of background substraction using cv2

import cv2

# Initialize the background subtractor using KNN (K-Nearest Neighbors) - More Robust Against Sudden Changes
# works well in dynamic lighting conditions.
# Similar to MOG2 but handles sudden changes better (e.g., switching lights on/off).
# Can remove background flickering.

bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# Open video stream
video_path="../testYolo/test2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    # Remove noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Removes small noise
    # Display original and foreground mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
