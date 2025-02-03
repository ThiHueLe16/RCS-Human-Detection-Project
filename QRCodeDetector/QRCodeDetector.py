import cv2
# Test if detectMarkers exists
print("detectMarkers exists:", hasattr(cv2.aruco, "detectMarkers"))

# Load the predefined ArUco dictionary for people and robots
aruco_dict_people = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # For people
aruco_dict_robot = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)  # For robots

# Create ArUco detector parameters
aruco_params = cv2.aruco.DetectorParameters()

# Open the video capture
cap = cv2.VideoCapture("../testYolo/test2.mp4")

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video ends

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers for both people and robots
    corners_people, ids_people, _ = cv2.aruco.detectMarkers(gray, aruco_dict_people, parameters=aruco_params)
    corners_robot, ids_robot, _ = cv2.aruco.detectMarkers(gray, aruco_dict_robot, parameters=aruco_params)

    # Draw detected markers for people
    if ids_people is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners_people, ids_people)

    # Draw detected markers for robots
    if ids_robot is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners_robot, ids_robot)

    # Display the frame with detected markers
    cv2.imshow("Detected Markers", frame)

    # Exit on pressing the 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
