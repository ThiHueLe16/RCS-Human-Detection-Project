import cv2
import numpy as np

from CircleBoundaryMonitor import CircleBoundaryMonitor


class ObjectDetection:
    @staticmethod
    def detect_objects(frame, center, monitor:CircleBoundaryMonitor):
        inner_radius = monitor.inner_radius
        outer_radius = monitor.outer_radius

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect objects (using threshold to turn the frame into binary framee or contour detection)
        _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        # Perform morphological operations to remove small noise (optional)
        kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel for dilation and erosion
        threshold = cv2.dilate(threshold, kernel, iterations=2)  # Dilate to connect potential contours
        threshold = cv2.erode(threshold, kernel, iterations=1)  # Erode to remove small noise
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the center of the object
            (x, y), radius = cv2.minEnclosingCircle(contour)
            object_center = (int(x), int(y))
            object_radius = int(radius)

            # Draw the detected object
            cv2.circle(frame, object_center, object_radius, (0, 255, 0), 2)
            cv2.circle(frame, object_center, 2, (255, 0, 0), -1)  # Mark object center

            # Calculate distance from the object to the circle's center
            distance_to_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Check if any part of the object is within the inner or outer radius
            if distance_to_center - object_radius <= inner_radius:
                # The object is inside the inner circle
                cv2.putText(frame, "ALARM: In danger Area!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("ALARM: Object inside inner circle!")
            # elif distance_to_center + object_radius >= outer_radius:
            #     # The object is outside the outer circle but breaching it
            #     cv2.putText(frame, "Caution: In caution area!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                 (0, 255, 255), 2)
            #     print("Caution: Object inside outer circle!")
            elif distance_to_center - object_radius <= outer_radius:
                # The object is outside the outer circle but breaching it
                cv2.putText(frame, "Caution: In caution area!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                print("Caution: Object inside outer circle!")
            else:
                cv2.putText(frame, "Safe", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                print("Safe")
        return frame
