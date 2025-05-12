from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

model = YOLO("yolov8n-oiv7.pt")  

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()        

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            confidence = conf.item()

            if confidence >= 0.6:
                x1, y1, x2, y2 = map(int, box.tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()