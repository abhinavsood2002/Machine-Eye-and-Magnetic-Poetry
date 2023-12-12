import cv2
import numpy as np
from collections import deque
import time
import os
from datetime import datetime
import platform
import requests

# Basic Camera Properties
CAPTURE_WIDTH = 640 # Resolution to capture W
CAPUTRE_HEIGHT = 480 # Resolution to capture H
CAPUTRE_RATE = 0.3 # Sample images every CAPTURE_RATE seconds. If there is an ANOMALY, capture at ANOMALY_CAPTURE_RATE

# Anomaly Related Parameters
ANOMALY_CAPTURE_RATE = 1 # Sample images every x seconds during an anomaly. These images will be storeqd
MINIMUM_ANOMALY_DURATION = 5 # Minimum anomaly duration in frames
MAXIMUM_ANOMALY_DURATION = 30
IMAGE_DIFFERENCE_BUFFER_LEN = 100 # Size of Buffer used to determine the if an image is an anomaly
ANOMALY_DIF_BUFFER_LEN = 10 # Size of Buffer used to determine if anomaly is no more significant
ANOMALY_SIGNIF = 5 # Frame triggers anomaly if frame_dif > ANOMALY_SIGIF * average frame_dif (image dif buffer) in buffer
ANOMALY_SIGNIF_END = 3 # Anomaly ends if frame_dif < ANOMALY_SIGIF_END * average frame_dif (anomaly dif buffer) in buffer
MINIMUM_TIME_BETWEEN_ANOMALY = 30 # An anomaly cannot be triggered to close to a previous one. (in frames)

# Parameters related to processing of frame difference
GUASSIAN_BLUR_KERNEL = (5,5) # Size of Guassian Blur for image difference
DILATION_EROSION_KERNEL = np.ones((5,5), np.uint8) # Size of Dilation/Erosion Kernel for image difference

def image_difference(image_1, image_2):
    image_1_bw = cv2.GaussianBlur(cv2.equalizeHist(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)), GUASSIAN_BLUR_KERNEL, 0)
    image_2_bw = cv2.GaussianBlur(cv2.equalizeHist(cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)), GUASSIAN_BLUR_KERNEL, 0)
    dilated_image_1 = cv2.dilate(image_1_bw, DILATION_EROSION_KERNEL, iterations=6)
    resulting_image_1 = cv2.erode(dilated_image_1, DILATION_EROSION_KERNEL, iterations=4)
    dilated_image_2 = cv2.dilate(image_2_bw, DILATION_EROSION_KERNEL, iterations=6)
    resulting_image_2 = cv2.erode(dilated_image_2, DILATION_EROSION_KERNEL, iterations=4)
    difference = np.power((resulting_image_1 - resulting_image_2), 2)
#    cv2.imshow("processed", resulting_image_1)
#
#    cv2.imshow("dif", difference)
    return np.sum(difference)

def start_program():
    # start webcam
    operating_system = platform.system()
    if operating_system == "Windows":
        print("Detected Windows")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    # print(cv2.getBuildInformation())
    if not cap.isOpened():
        print("Error: failed to open video stream")
    else:
        print("Video Camera Loaded")
    # Caputure at 480p
    cap.set(3, CAPTURE_WIDTH)
    cap.set(4, CAPUTRE_HEIGHT)

    # Instead of setting constant exposure, use Histogram Equalisation
    # cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    # model
    # model = YOLO("models/yolov8/yolov8n.pt")

    # object classes
    # classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    #               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    #               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    #               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    #               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    #               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    #               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    #               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    #               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    #               "teddy bear", "hair drier", "toothbrush"
    #               ]



    buffer_img_dif = deque(maxlen=IMAGE_DIFFERENCE_BUFFER_LEN)

    prev_img_time = time.time()
    anomaly_mode = False
    prev_img = None
    anomalies_written = 0
    while True:
        image_time = time.time()
        # Normal Sampling
        if image_time - prev_img_time > 0.3 and not anomaly_mode:
            success, img = cap.read()
            print("Sampled Image")
            if prev_img is not None:
                frame_difference = image_difference(img, prev_img)
                buffer_img_dif.append(frame_difference)
                if frame_difference > ANOMALY_SIGNIF * np.average(buffer_img_dif) and len(buffer_img_dif) > 20:
                    print("ACTIVATING ANOMALY MODE")
                    if not os.path.exists("images"):
                        os.makedirs("images")
                    current_datetime = datetime.now()
                    anomaly_name = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
                    folder_name = "images" + "/" + anomaly_name + "/"
                    os.mkdir(folder_name)
                    os.chdir(folder_name)
                    cv2.imwrite(str(anomalies_written) + ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
                    anomalies_written += 1
                    anomaly_mode = True

#            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

            prev_img = img
            prev_img_time = image_time

        # Anomaly Detected
        elif anomaly_mode:
            prev_anomaly_time = time.time()
            prev_anomaly_img = None
            anomaly_buffer = deque(maxlen=ANOMALY_DIF_BUFFER_LEN)
            anomaly_done = False
            buffer_updates = 0
            while True:
                anomaly_time = time.time()
                if anomaly_done and anomaly_time - prev_anomaly_time > CAPUTRE_RATE:
                    print("Anomaly Mode - Buffer Update")
                    # Continue updating Image dif buffer to ensure wasteful anomalies are not captured later
                    success, img = cap.read()
                    frame_difference = image_difference(img, prev_anomaly_img)
                    buffer_img_dif.append(frame_difference)
                    prev_anomaly_img = img
                    prev_anomaly_time = anomaly_time
                    buffer_updates += 1
                    if buffer_updates > (MINIMUM_TIME_BETWEEN_ANOMALY):
                        print("EXITING ANOMALY MODE")
                        anomalies_written = 0
                        break

                elif anomaly_time - prev_anomaly_time > ANOMALY_CAPTURE_RATE:
                    success, img = cap.read()
                    print("Anomaly Mode - Sampled Image")
                    cv2.imwrite(str(anomalies_written) + ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
                    if anomalies_written < MINIMUM_ANOMALY_DURATION:
                        if prev_anomaly_img is not None:
                            frame_dif = image_difference(img, prev_anomaly_img)
                            anomaly_buffer.append(frame_dif)
                    else:
                        frame_dif = image_difference(img, prev_anomaly_img)
                        if frame_dif < ANOMALY_SIGNIF_END * np.average(anomaly_buffer) or anomalies_written > MAXIMUM_ANOMALY_DURATION:
                            os.chdir("..")
                            os.chdir("..")
                            anomaly_done = True
                            anomalies_written = 0


                    prev_anomaly_time = anomaly_time
                    prev_anomaly_img = img
                    anomalies_written += 1

            # prev_anomaly_img

            anomaly_mode = False
    cap.release()
    cv2.destroyAllWindows()
start_program()