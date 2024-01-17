# MachineEye and Magnetic Poetry
This repository contains the code regarding the development of the Machine Eye system. The multiple components
in the machine system are recorded in various branches of this repository. The current branches implementation in the repository are as follows:

**Machine Eye Related**
- **rpi**: This branch contains the code that is run on a Raspberry Pi (Model 4B) with a Luxonis Oak-D lite camera.
- **server**: This branch contains the implementation of the remote server that stores and processes information collected on the Raspberry Pi.

**Magnetic Poetry Related**
- **poetry-rpi**: This branch contains the code that is run on a Raspberry Pi (Model 4B) with a  Luxonis Oak-D lite camera. 
- **poetry-server**: This branch contains the implementation of the remote server that stores and processes information collected on the Raspberry Pi.

## How the system works

### Machine Eye
- The RPI code is run, a socket is created and waits for a connection
- The server code is run, and the socket on the RPI is connected with
- The Luxonis camera starts capturing pictures at its fps
- When people are detected in the picture using Yolo, the number of people is added to an exponentially weighted moving average buffer (to prevent erroneous detections from being saved and to detect anomalies).
- This is also done for objects detected.
- When either of the values that we get from the buffer, exceeds a manually defined significance threshold, we start processing an anomaly
- In an anomaly, we send a given number of images to the machine eye server to save via the sockets defined before
  
### Magnetic Poetry
- The server code is run, a socket is created and waits for a connection
- The RPI code is run, and the socket on the server is connected with
- The Luxonis camera starts capturing pictures at its fps and runs the corresponding pipeline
- Once the palm model on the camera detects a hand, the number detected, is added to an Exponentially Weighted moving average buffer.
- If the value in the buffer exceeds a given threshold, the local ongoing variable is set to true.
- Once the value drops below the threshold, an image is captured, cropped (The original 4K image is too expensive to send) and sent to the server via a socket.
- On the server, the image is received and EasyOCR is run on a sharpened and rotated version of the image to get the text in the image.
- Using the bounding boxes of each magnet detected, the magnets are ordered. (This needs to be refined)


This contains observations that may/may not be useful for anyone who continues to choose to work on the developed system. These logs contain basic observations about different components and "shallow" recordings of the work done.
