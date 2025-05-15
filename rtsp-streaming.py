from picamera2 import Picamera2
from flask import Flask, Response
import cv2
import threading
import numpy as np
from ultralytics import YOLO
import os
from queue import Queue

app = Flask(__name__)

# Setup FIFO path for RTSP streaming
fifo_path = "/tmp/yolo_stream.mjpg"
if not os.path.exists(fifo_path):
    os.mkfifo(fifo_path)

# Initialize Picamera2 and configure video stream
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (320, 240)}))
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Load YOLO model
model = YOLO("yolov8n-oiv7.pt")

# Shared frame buffer and queue
frame_queue = Queue(maxsize=1)


def capture_frames():
    while True:
        temp_frame = picam2.capture_array()
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(temp_frame)


def process_frames():
    fifo = open(fifo_path, 'wb')
    while True:
        if not frame_queue.empty():
            temp_frame = frame_queue.get()
            results = model(temp_frame)
            for result in results:
                frame = result.plot()
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    fifo.write(jpeg.tobytes())
                    fifo.flush()


@app.route('/video_feed')
def video_feed():
    # Optional: you can still serve MJPEG on HTTP if you want (same as before)
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