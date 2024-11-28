import cv2
import numpy as np

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

    motion_mask = (magnitude > 2.0).astype(np.uint8) * 255  # Adjust threshold if needed, higher threshold to reduce sensitive to really small, insignificant cauuse by shadow or lighting
    return motion_mask

# Initialize video capture
video_path = "../testYolo/peopleTestYolo.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

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

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect motion between the previous and current frame
    motion_mask = detect_motion_with_optical_flow(prev_gray, gray)

    # Display the motion mask
    cv2.imshow("Motion Mask", motion_mask)

    # Display the original frame with motion overlay
    colored_mask = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
    cv2.imshow("Motion Detection", overlay)

    # Update the previous frame
    prev_gray = gray

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
