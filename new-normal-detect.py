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
        frame = result.plot()

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()