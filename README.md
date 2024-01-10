# MachineEye - Server
This repository contains the code regarding the development of the Machine Eye system on the remote host (currently the ml station at SensiLab). The current implementation is very simple and just contains sockets for saving images from the rpi. The postprocessing folder contains post-processing scripts that are applied to the images collected for further filtering of input. They are not currently used in the current code base and were initially used with the old vision system to reduce False positives.

# Setup
Create a virtual environment with packages corresponding to the requirements.txt file. Then run ```python start.py``` after the rpi is already running

