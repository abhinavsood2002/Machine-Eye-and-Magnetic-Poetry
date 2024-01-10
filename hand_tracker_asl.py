from PIL import ImageFont
import numpy as np
import copy
import itertools
import collections
from collections import namedtuple
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv
import os 
import socket
import pickle

load_dotenv("./configs/standard_ml_station.env")
characters = ['A', 'B', 'C', 'D', 
              'E', 'F', 'G', 'H', 
              'I', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 
              'R', 'S', 'T', 'U', 
              'V', 'W', 'X', 'Y']

FINGER_COLOR = [(128, 128, 128), (80, 190, 168), 
         (234, 187, 105), (175, 119, 212), 
         (81, 110, 221)]

JOINT_COLOR = [(0, 0, 0), (125, 255, 79), 
            (255, 102, 0), (181, 70, 255), 
            (13, 63, 255)]

# Choose font, scale, color, and thickness
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
color = (255, 255, 255)  # White color in BGR
thickness = 2

# Server configuration
PROCESSING_SERVER_IP = os.getenv("PROCESSING_SERVER_IP") # Server IP address
PROCESSING_SERVER_PORT = int(os.getenv("PROCESSING_SERVER_PORT")) # Port to listen on

# Crop parameters
H_START = int(os.getenv("H_START"))
H_END = int(os.getenv("H_END"))
V_START = int(os.getenv("V_START"))
V_END = int(os.getenv("V_END"))

# Hyperparameters related to compu((ter vision
WAIT_AFTER_HANDS_LEAVE = float(os.getenv("WAIT_AFTER_HANDS_LEAVE"))
HANDS_DETECTED_THRESHOLD = float(os.getenv("HANDS_DETECTED_THRESHOLD"))
HANDS_LEFT_THRESHOLD = float(os.getenv("HANDS_LEFT_THRESHOLD"))
ALPHA = float(os.getenv("ALPHA"))

PALM_DETECTION_THRESH = float(os.getenv("PALM_DETECTION_THRESH"))
PALM_DETECTION_NMS = float(os.getenv("PALM_DETECTION_NMS"))

num_hands_buffer = mpu.ExponentialWeightedMovingAverage(ALPHA)
# Define socket to transfer images
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Server is waiting for a connection...")
client_socket.connect((PROCESSING_SERVER_IP, PROCESSING_SERVER_PORT))
print(f"Connected to Server")

# Send an image via a socket
def send_image(conn, folder_name, image_data):
    
    image_serialized = pickle.dumps(image_data)
    print(folder_name)
    
    # Send folder name length and folder name as bytes
    folder_name_bytes = folder_name.encode()
    conn.sendall(len(folder_name_bytes).to_bytes(2, byteorder='big'))
    conn.sendall(folder_name_bytes)

    # Send image size
    image_size = len(image_serialized)
    conn.sendall(image_size.to_bytes(4, byteorder="big"))

    # Send image
    conn.sendall(image_serialized)

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
    return resized

class HandTrackerASL:
    def __init__(self,
                pd_path="models/palm_detection_6_shaves.blob", 
                pd_score_thresh=PALM_DETECTION_THRESH, pd_nms_thresh=PALM_DETECTION_NMS,
                lm_path="models/hand_landmark_6_shaves.blob",
                lm_score_threshold=0.5,
                show_landmarks=True,
                show_hand_box=True,
                asl_path="models/hand_asl_6_shaves.blob",
                asl_recognition=True,
                show_asl=True):

        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.asl_path = asl_path
        self.show_landmarks=show_landmarks
        self.show_hand_box = show_hand_box
        self.asl_recognition = asl_recognition
        self.show_asl = show_asl
        self.hands_detection_context = {"ongoing": False}

        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        self.frame_size = None

        self.right_char_queue = collections.deque(maxlen=5)
        self.left_char_queue = collections.deque(maxlen=5)
        
        self.preview_width = 576
        self.preview_height = 324
        
        # Cropping for preview_frame, Height and Width from 4K res
        self.preview_frame_crop_H = self.preview_width / 3840
        self.preview_frame_crop_V = self.preview_height / 2160 
        
        self.previous_right_char = ""
        self.right_sentence = ""
        self.previous_right_update_time = time.time()
        self.previous_left_char = ""
        self.left_sentence = ""
        self.previous_left_update_time = time.time()


    def create_pipeline(self):
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        self.pd_input_length = 128

        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setInterleaved(False)
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.initialControl.setSharpness(1)     # range: 0..4, default: 1
        cam.initialControl.setLumaDenoise(1)   # range: 0..4, default: 1
        cam.initialControl.setChromaDenoise(4) 
        
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.video.link(cam_out.input)

        cam_nn_out = pipeline.createXLinkOut()
        cam_nn_out.setStreamName("cam_nn_out")
        cam.preview.link(cam_nn_out.input)

        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        pd_in = pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        print("Pipeline created.")
        return pipeline


    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        mpu.detections_to_rect(self.regions)
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)


    def process_hand_detection(self, hands_detection_average, frame = None):
        if  hands_detection_average > HANDS_LEFT_THRESHOLD:
            self.hands_detection_context["ongoing"] = True
        else:
            time.sleep(WAIT_AFTER_HANDS_LEAVE)
            self.hands_detection_context["ongoing"] = False
            if frame is not None:
                current_datetime = datetime.now()
                folder_name = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
                cv2.imwrite("images/" + folder_name + ".png", frame)
                send_image(client_socket, folder_name, frame)

    def run(self):
        device = dai.Device(self.create_pipeline())
        q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        q_preview = device.getOutputQueue(name="cam_nn_out", maxSize=1, blocking=False)
        q_pd_in = device.getInputQueue(name="pd_in")
        q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
    
        while True:
            in_video = q_video.get()
            in_preview = q_preview.get()
            video_frame = in_video.getCvFrame()
            preview_frame = in_preview.getCvFrame()

            # Crop using Determine Utilities tool
            to_send = video_frame[V_START:V_END, H_START:H_END]

            h, w = preview_frame.shape[:2]
            self.frame_size = max(h, w)
            self.pad_h = int((self.frame_size - h)/2)
            self.pad_w = int((self.frame_size - w)/2)
            preview_frame = cv2.copyMakeBorder(preview_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

            frame_nn = dai.ImgFrame()
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(to_planar(preview_frame, (self.pd_input_length, self.pd_input_length)))
            q_pd_in.send(frame_nn)
          
            # cv2.imshow("nn_frame", preview_frame)
            # cv2.waitKey(1)

            # Get palm detection
            inference = q_pd_out.get()
            self.pd_postprocess(inference)
            num_hands_buffer.add_value(len(self.regions))
            average = num_hands_buffer.get_average()
            if average > HANDS_DETECTED_THRESHOLD and not self.hands_detection_context["ongoing"]:
                self.hands_detection_context["ongoing"] = True
                self.process_hand_detection(average)
                print("hands detected")
            if self.hands_detection_context["ongoing"]:
                self.process_hand_detection(average, to_send)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd_m", default="models/palm_detection_6_shaves.blob", type=str,
                        help="Path to a blob file for palm detection model (default=%(default)s)")
    parser.add_argument("--lm_m", default="models/hand_landmark_6_shaves.blob", type=str,
                        help="Path to a blob file for landmark model (default=%(default)s)")
    parser.add_argument("--asl_m", default="models/hand_asl_6_shaves.blob", type=str,
                        help="Path to a blob file for ASL recognition model (default=%(default)s)")
    parser.add_argument('-asl', '--asl', default=True, 
                        help="enable ASL recognition")
    args = parser.parse_args()
    
    try:
        ht = HandTrackerASL(pd_path=args.pd_m, lm_path=args.lm_m, asl_path=args.asl_m, asl_recognition=args.asl)
        ht.run()
    except Exception as e:
        print(e)
        client_socket.close()
