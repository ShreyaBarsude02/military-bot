from picamera2 import Picamera2
from flask import Flask, Response
import cv2
import threading
import numpy as np
from ultralytics import YOLO
import os
from queue import Queue

app = Flask(__name__)

fifo_path = "/tmp/yolo_stream.mjpg"
if not os.path.exists(fifo_path):
    os.mkfifo(fifo_path)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (320, 240)}))
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

model = YOLO("yolov8n-oiv7.pt")

frame_queue = Queue(maxsize=1)


def capture_frames():
    while True:
        temp_frame = picam2.capture_array()
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(temp_frame)


def process_frames():
    global frame
    while True:
        if not frame_queue.empty():
            temp_frame = frame_queue.get()

            bgr_frame = cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR)

            results = model(bgr_frame)

            for result in results:
                frame = result.plot()


@app.route('/video_feed')
def video_feed():

    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=process_frames, daemon=True).start()

    app.run(host='0.0.0.0', port=8000)