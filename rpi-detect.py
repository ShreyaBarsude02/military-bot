from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n-oiv7.pt")  

# Define your custom risk categories
high_risk_classes = [
    'Aircraft', 'Airplane', 'Axe', 'Bomb', 'Bow and arrow', 'Cannon', 'Fireplace', 'Gas stove',
    'Hammer', 'Handgun', 'Helicopter', 'Knife', 'Missile', 'Rifle', 'Suitcase', 'Sword', 'Tank', 'Truck', 'Van'
]

medium_risk_classes = [
    'Backpack', 'Ball', 'Beaker', 'Belt', 'Bicycle', 'Bicycle helmet', 'Bicycle wheel', 
    'Boot', 'Box', 'Camera', 'Container', 'Glove', 'Torch'
]

# Colors
COLOR_HIGH = (0, 0, 255)      # Red
COLOR_MEDIUM = (0, 255, 255)  # Yellow
COLOR_NORMAL = (0, 255, 0)    # Green

# Get class names from model
class_names = model.names  # {0: 'person', 1: 'bicycle', ...}

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()        

    results = model(frame)[0]  # Get first (and only) result

    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        confidence = conf.item()
        class_id = int(cls.item())
        class_name = class_names[class_id]

        if confidence >= 0.6:
            x1, y1, x2, y2 = map(int, box.tolist())

            # Decide color based on risk class
            if class_name in high_risk_classes:
                color = COLOR_HIGH
            elif class_name in medium_risk_classes:
                color = COLOR_MEDIUM
            else:
                color = COLOR_NORMAL  # Default green for normal

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add location text at top right
    cv2.putText(frame, "Location: VIT Pune", (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
