import cv2
import numpy as np

# Define real-world points for a 6x6m working area (corners of the square)
real_world_points = np.array([
    [-3, -3],  # Bottom-left corner
    [3, -3],   # Bottom-right corner
    [3, 3],    # Top-right corner
    [-3, 3]    # Top-left corner
], dtype=np.float32)

# Define corresponding points in the camera frame (pixel coordinates from calibration)
camera_frame_points = np.array([
    [320, 400],  # Bottom-left corner in the camera frame
    [640, 400],  # Bottom-right corner in the camera frame
    [640, 100],  # Top-right corner in the camera frame
    [320, 100]   # Top-left corner in the camera frame
], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(camera_frame_points, real_world_points)

# Initialize video capture
cap = cv2.VideoCapture("your_video_file.mp4")  # Replace with your video file or camera ID

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Assume you have a detection (bounding box) from your model
    # Example: A detection point in the camera frame
    detected_point_camera = np.array([[450, 300]], dtype=np.float32)  # Example detection (x, y)
    detected_point_camera = np.array([detected_point_camera])  # Add a dimension for homography

    # Transform the point to the real-world top-down view
    transformed_point = cv2.perspectiveTransform(detected_point_camera, H)

    # Extract x, y for plotting
    x, y = transformed_point[0][0]

    # Draw the top-down map
    top_down_map = np.zeros((500, 500, 3), dtype=np.uint8)  # Create a blank image
    center = (250, 250)  # Center of the cobot in the top-down view

    # Define scaling: 1 meter = 50 pixels (6m x 6m working area fits within 500x500 pixels)
    scale = 50

    # Draw the danger and caution zones
    cv2.circle(top_down_map, center, int(2 * scale), (0, 0, 255), 2)  # Red circle (Danger Zone)
    cv2.circle(top_down_map, center, int(3 * scale), (0, 255, 255), 2)  # Yellow circle (Caution Zone)

    # Transform the coordinates to the top-down map
    point_on_map = (int(center[0] + x * scale), int(center[1] - y * scale))  # Adjust to pixel coordinates
    cv2.circle(top_down_map, point_on_map, 5, (255, 0, 0), -1)  # Draw the detected point

    # Show the original frame and the top-down map
    cv2.imshow("Camera Frame", frame)
    cv2.imshow("Top-Down View", top_down_map)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
