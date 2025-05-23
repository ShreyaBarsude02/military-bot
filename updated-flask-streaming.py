from picamera2 import Picamera2
from flask import Flask, Response
import cv2
import threading
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
model = YOLO("yolov8n-oiv7.pt")

frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while True:
        temp_frame = picam2.capture_array()
        results = model(temp_frame)

        for result in results:
            plotted_frame = result.plot()

        with lock:
            frame = plotted_frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.close()
    cv2.destroyAllWindows()
    os._exit(0)

@app.route('/video_feed')
def video_feed():
    def generate():
        global frame
        while True:
            with lock:
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                else:
                    continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=8000)
