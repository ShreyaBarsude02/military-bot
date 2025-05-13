from picamera2 import Picamera2
from flask import Flask, Response
import cv2
import threading
from ultralytics import YOLO
import numpy as np
import os

app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
model = YOLO("yolov8n-oiv7.pt") 
frame = None

def capture_frames():
    global frame
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
    os._exit(0)  # Stop Flask too

@app.route('/video_feed')
def video_feed():
    def generate():
        global frame
        while True:
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=8000)

#http://192.168.0.105:8000/video_feed