#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include "httplib.h"  // Add this header from cpp-httplib

using namespace cv;
using namespace dnn;
using namespace std;

mutex frame_mutex;
Mat latest_frame;
bool running = true;

// YOLO parameters
const float confThreshold = 0.25;
const float nmsThreshold = 0.45;
const int inputWidth = 640;
const int inputHeight = 640;
vector<string> classNames;

// GStreamer pipeline for Pi camera
string gstreamer_pipeline(int capture_width, int capture_height, int framerate, int display_width, int display_height) {
    return "libcamerasrc ! "
           "video/x-raw, width=(int)" + to_string(capture_width) +
           ", height=(int)" + to_string(capture_height) +
           ", framerate=(fraction)" + to_string(framerate) + "/1 ! "
           "videoconvert ! videoscale ! "
           "video/x-raw, width=(int)" + to_string(display_width) +
           ", height=(int)" + to_string(display_height) + " ! appsink";
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
    string label = format("%.2f", conf);
    if (!classNames.empty()) {
        CV_Assert(classId < (int)classNames.size());
        label = classNames[classId] + ": " + label;
    }
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(), 1);
}

void postprocess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    float* data = (float*)outs[0].data;
    const int dimensions = 85;
    const int rows = outs[0].rows;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= confThreshold) {
            float* classesScores = data + 5;
            Mat scores(1, 80, CV_32FC1, classesScores);
            Point classIdPoint;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

            if (maxClassScore > confThreshold) {
                confidences.push_back(confidence * (float)maxClassScore);
                classIds.push_back(classIdPoint.x);

                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((cx - 0.5 * w) * frame.cols / inputWidth);
                int top = int((cy - 0.5 * h) * frame.rows / inputHeight);
                int width = int(w * frame.cols / inputWidth);
                int height = int(h * frame.rows / inputHeight);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        data += dimensions;
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void camera_loop() {
    string pipeline = gstreamer_pipeline(640, 480, 30, 640, 480);
    VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Failed to open camera using GStreamer pipeline!\n";
        running = false;
        return;
    }

    Net net = readNetFromONNX("yolov8n.onnx");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat frame;
    while (running) {
        cap >> frame;
        if (frame.empty()) continue;

        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        postprocess(frame, outputs);

        {
            lock_guard<mutex> lock(frame_mutex);
            latest_frame = frame.clone();
        }

        this_thread::sleep_for(chrono::milliseconds(30));
    }
}

int main() {
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)) classNames.push_back(line);

    thread cam_thread(camera_loop);

    httplib::Server svr;

    svr.Get("/video_feed", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame");

        while (running) {
            Mat frame_copy;
            {
                lock_guard<mutex> lock(frame_mutex);
                if (latest_frame.empty()) continue;
                frame_copy = latest_frame.clone();
            }

            vector<uchar> buff;
            imencode(".jpg", frame_copy, buff);

            res.write("--frame\r\n");
            res.write("Content-Type: image/jpeg\r\n\r\n");
            res.write(reinterpret_cast<char*>(buff.data()), buff.size());
            res.write("\r\n");

            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    cout << "Server running at http://localhost:8080/video_feed\n";
    svr.listen("0.0.0.0", 8080);

    running = false;
    cam_thread.join();

    return 0;
}