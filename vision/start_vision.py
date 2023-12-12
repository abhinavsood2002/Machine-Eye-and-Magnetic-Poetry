#!/usr/bin/env python3
from pathlib import Path
from collections import deque
import cv2
import depthai as dai
import numpy as np
import time
import os
from datetime import datetime
from util import averages
import socket
from dotenv import load_dotenv
import pickle

load_dotenv(dotenv_path="./configs/standard_ml_station.env")

# Server configuration
PROCESSING_SERVER_IP = os.getenv("PROCESSING_SERVER_IP") # Server IP address
PROCESSING_SERVER_PORT = int(os.getenv("PROCESSING_SERVER_PORT")) # Port to listen on

PEOPLE_SIGNIFICANCE = float(os.getenv("PEOPLE_SIGNIFICANCE")) # Anomaly triggers if number of people is x times more than average in buffer
OBJECT_SIGNIFICANCE = float(os.getenv("OBJECT_SIGNIFICANCE")) # Anomaly triggers if number of objects is x times more than average in buffer

# Instead of storing a buffer we maintain a exponential weighted moving average
# The buffer alpha determines how much emphasis of the average is placed on recent events
# Higher the alpha the more emphasis on recent values in the average
BUFFER_ALPHA = float(os.getenv("BUFFER_ALPHA"))

ANOMALY_DURATION = int(os.getenv("ANOMALY_DURATION")) # anomaly duration in number of frames
ANOMALY_FPS = float(os.getenv("ANOMALY_FPS")) # anomaly caputer number of frames per second
MINIMUM_TIME_BETWEEN_ANOMALY = float(os.getenv("MINIMUM_TIME_BETWEEN_ANOMALY")) # in seconds

# The model file was originally a .pt file and converted to .blob using https://tools.luxonis.com/
nnPath = str((Path(__file__).parent / Path(os.getenv("MODEL_PATH"))).resolve().absolute())

# Define socket to transfer images
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((PROCESSING_SERVER_IP, PROCESSING_SERVER_PORT))
server_socket.listen()
print("Server is waiting for a connection...")
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

# Send an image via a socket
def send_image(conn, folder_name, image_name, image_data):
    image_serialized = pickle.dumps(image_data)
    print(folder_name)
    
    # Send folder name length and folder name as bytes
    folder_name_bytes = folder_name.encode()
    conn.sendall(len(folder_name_bytes).to_bytes(2, byteorder='big'))
    conn.sendall(folder_name_bytes)

    # Send image name length and image name as bytes
    image_name_bytes = image_name.encode()
    conn.sendall(len(image_name_bytes).to_bytes(2, byteorder='big'))
    conn.sendall(image_name_bytes)

    # Send image size
    image_size = len(image_serialized)
    conn.sendall(image_size.to_bytes(4, byteorder="big"))

    # Send image
    conn.sendall(image_serialized)

# YOLO labels
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutPreview = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
xoutPreview.setStreamName("preview")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(416, 416)    # NN input
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setPreviewKeepAspectRatio(False) # Strech images for yolov8
camRgb.setIspScale(5, 10) # Downsize 4K output for easier processing

# Network specific settings
detectionNetwork.setConfidenceThreshold(0.4)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.video.link(xoutVideo.input)
camRgb.preview.link(xoutPreview.input)
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.out.link(nnOut.input)

people_buffer = averages.ExponentialWeightedMovingAverage(alpha=BUFFER_ALPHA)
objects_buffer = averages.ExponentialWeightedMovingAverage(alpha=BUFFER_ALPHA)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the frames and nn data from the outputs defined above
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    qPreview = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    previewFrame = None
    videoFrame = None
    detections = []
    
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    def process_anomaly():

        print("ACTIVATING ANOMALY MODE")
        if not os.path.exists("images"):
            os.makedirs("images")
        current_datetime = datetime.now()
        anomaly_name = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
        folder_name = "images" + "/" + anomaly_name + "/"
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for i in range(ANOMALY_DURATION):
            image = qVideo.get().getCvFrame()
            cv2.imwrite(str(i) + ".png", image)
            send_image(conn, anomaly_name, str(i), image)
            time.sleep(1/ANOMALY_FPS)
        os.chdir("../..")
        

    #cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("video", 1280, 720)
    print("Resize video window with mouse drag!")
    prev_anomaly_time = time.time()
    while True:

        inVideo = qVideo.tryGet()
        inPreview = qPreview.tryGet()
        inDet = qDet.tryGet()

        if inVideo is not None:
            videoFrame = inVideo.getCvFrame()

        if inPreview is not None:
            previewFrame = inPreview.getCvFrame()

        if inDet is not None:
            detections = inDet.detections

        if videoFrame is not None:
            # displayFrame("video", videoFrame)
            pass
        if detections is not None:
            num_people = sum([labelMap[detection.label] == "person" for detection in detections])
            num_objects = len(detections) - num_people
            people_buffer.add_value(num_people)
            objects_buffer.add_value(num_objects)
            time_from_last = time.time() - prev_anomaly_time
            if ((num_people > people_buffer.get_average() * PEOPLE_SIGNIFICANCE or 
               num_objects > objects_buffer.get_average() * OBJECT_SIGNIFICANCE) and
               time_from_last > MINIMUM_TIME_BETWEEN_ANOMALY):
                process_anomaly()
                prev_anomaly_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            server_socket.close()
            break
