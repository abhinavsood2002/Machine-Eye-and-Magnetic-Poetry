# MachineEye - RPI implementation
This repository contains the code implementation that runs on the Raspberry Pi.

## Vision System
The vision system has gone through multiple stages of improvement. At the current stage, it utilises object detection with yolov8 nano running on a Luxonis OAK-D Lite camera. Images with activity are then sent via a socket connected to a remote server for storage and further processing.

### Legacy vision
The legacy folder contains an alternative implementation of the vision system that uses frame difference to capture anomalies in _old_start.py_. _buggy_luxonis_mix.py_ contains an implementation of feature tracking and object detection combined that currently crashes on the luxonis camera due to memory issues regarding the camera pipeline freezing.

## Setup
Create a virtual environment with the libraries in requirements.txt and run ```python vision\start_vision.py ``` from the MachineEye directory. The system has been tested on **Python 3.10.6**.

### Configuration
To transfer files automatically from the rpi to the remote server the correct variables need to be set in ```configs/standard_ml_station.env```. Additional parameters exist that affect the accuracy and sensitivity of the vision system
