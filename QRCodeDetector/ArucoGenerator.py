import cv2
import numpy as np

# Define the dictionary (e.g., DICT_4X4_50, DICT_5X5_100, etc.)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # 4x4 marker with 50 IDs

# Loop to generate multiple markers (IDs 0 to 9)
for marker_id in range(10):
    marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, 200)  # 200 is the side length
    filename = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(filename, marker_image)
    print(f"Saved marker {marker_id} as {filename}")
