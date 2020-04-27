YOLOv3 COCO Inference Web Service for Intel® Movidius NCS2
==========================================================

<img src="doc/example-1.png" width="700">

Intro
--------
This project is made to run on a Raspberry Pi, though it should run on any linux device that has OpenVINO and NCS2 support.


Dependencies
------------

coco.names<br>
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
<p>
frozen_darknet_yolov3_model.xml<br>
frozen_darknet_yolov3_model.bin<br>
<p>
YOLOv3 Darknet model (trained on coco dataset) in IR format for Intel Movidius.<br>
original models: https://pjreddie.com/darknet/yolo/
<p>
Tutorial to convert models to Intel Movidius IR
http://www.keymolen.com/2020/04/run-yolov3-on-raspberry-pi-with-intel.html


Run
---
python3 movidius-inference-server.py
