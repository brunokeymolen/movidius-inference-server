YoloV3 Intel Movidius NCS2 Inference Server
===========================================

<img src="doc/example-1.png" width="700">

Tutorial:
--------
http://www.keymolen.com


Dependencies:
------------

coco.names<br>
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
<p>
frozen_darknet_yolov3_model.xml<br>
frozen_darknet_yolov3_model.bin<br>
<p>
These is a converted YoloV3 Darknet model to Intel Movidius IR.<br>
original models: https://pjreddie.com/darknet/yolo/
<p>
Tutorial to convert models for Raspberry Pi:<br>
see http.keymolen.com/<br>


Run
---
python3 movidius-inference-server.py
