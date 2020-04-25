YoloV3 Intel Movidius NCS2 Inference Server
===========================================

<img src="doc/example-1.png" width="700">

Tutorial:
--------
http://www.keymolen.com


Dependencies:
------------

coco.names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

frozen_darknet_yolov3_model.xml
frozen_darknet_yolov3_model.bin

These is a converted YoloV3 Darknet model to Intel Movidius IR.
original models: https://pjreddie.com/darknet/yolo/

Tutorial to convert models for Raspberry Pi:
see http.keymolen.com/ 


Run
---
python3 movidius-inference-server.py
