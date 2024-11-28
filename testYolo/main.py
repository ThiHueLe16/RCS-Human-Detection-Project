import cv2
from ultralytics import YOLO
import numpy as np
import torch



# class specification of yolo8
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

# Initialize video capture and YOLO model
cap=cv2.VideoCapture("peopleTestYolo.mp4")
model=YOLO("yolov8m.pt")

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame=cap.read()
    if not ret:
        break
    # YOLOv8 object detection
    results= model(frame)
    result=results[0]
    boundingboxes= np.array(result.boxes.xyxy.cpu(),dtype="int")
    classes= np.array(result.boxes.cls.cpu(),dtype="int")
    # print(boundingboxes)

    # Apply background subtraction to detect motion
    fg_mask = backSub.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask_thresh = cv2.erode(fg_mask_thresh, None, iterations=2)
    fg_mask_thresh = cv2.dilate(fg_mask_thresh, None, iterations=2)

    # Find contours for motion detection
    contours, _ = cv2.findContours(fg_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for class_,box in zip(classes,boundingboxes):
        (x,y, x2, y2)=box
        cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255),2)
        cv2.putText(frame, str(names[class_]),(x,y-5), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # Draw bounding boxes for motion
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Ignore small movements
                x, y, w, h = cv2.boundingRect(contour)
                overlap = False

                # Check if the motion overlaps with YOLO boxes
                for box in boundingboxes:
                    bx, by, bx2, by2 = box
                    if (x < bx2 and x + w > bx and y < by2 and y + h > by):  # Intersection condition
                        overlap = True
                        break

                if not overlap:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Undefined", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display
    cv2.imshow("Image", frame)
    key=cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()

