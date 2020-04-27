#
# movidius-inference-server
# Copyright (C)2020 - Bruno Keymolen, bruno.keymolen@gmail.com
#
# Inspiration and some code fragments are from the Intel Python Demo:
# https://github.com/opencv/open_model_zoo/blob/master/demos/python_demos/object_detection_demo_yolov3_async/object_detection_demo_yolov3_async.py
# 
# Used model: 
# Darknet YoloV3, COCO Dataset: https://pjreddie.com/darknet/yolo/
#
# Tutorial to this source code:
# https://www/keymolen.com
#

import cv2 as cv
import time
import numpy as np
import math
import time
import io
import os
import uuid

from flask import Flask, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
app = Flask(__name__)



threshold_pred_box = 0.3
threshold_conf = 0.65
threshold_nms = 0.4

net = None
classes = None

current_milli_time = lambda: int(round(time.time() * 1000))

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#return: confidence, class-id
def processouts(frame, outs):
    yolov3_anchors = { 52:[[10,13],[16,30],[33,23]], 26:[[30,61],[62,45],[59,119]], 13:[[116,90],[156,198],[373,326]] }
    result = []
    boxes = []
    confidences = []
    classids = []
    frame_w = frame.shape[2]
    frame_h = frame.shape[3]
    for out in outs:
        #example: out.shape (1, 255, 26, 26)  [0][c][y][x]
        cx = out.shape[2]
        cy = out.shape[3]
        anchors = yolov3_anchors[cx] #cx == cy
        for x in range(cx):
            for y in range(cy):
                for pred in range(3):
                    #x,y,w,h,P,c[80] (x3)
                    conf = out[0][4+(pred*85)][y][x]  
                    if conf > threshold_pred_box:
                        # get the class
                        mx = 0.0;
                        mxi = 0;
                        for c in range(80):
                            cprob = out[0][5+(pred*85)+c][y][x]
                            if cprob > mx:
                                mx = cprob
                                mxi = c

                        if mx >= threshold_conf:
                            #print("objectness score ({}) {}x{} (cells {}x{}) = {} class index: {} name: {} prob: {}"
                            #        .format(pred,x,y,cx,cy,conf, mxi, classes[mxi], mx))
                            
                            #Process the box
                            tx = out[0][0+(pred*85)][y][x]
                            ty = out[0][1+(pred*85)][y][x]
                            tw = out[0][2+(pred*85)][y][x]
                            th = out[0][3+(pred*85)][y][x]
                           
                            #Center
                            center_x = (frame_w/cx) * (x + tx)
                            center_y = (frame_h/cy) * (y + ty) 
                            
                            #https://pjreddie.com/media/files/papers/YOLOv3.pdf
                            #bw = Pw * e^tw 
                            #bh = Ph * e^th
                            try:
                                w_exp = math.exp(tw)         
                                h_exp = math.exp(th)         
                            except OverflowError:
                                continue
                            
                            box_w = anchors[pred][0] * w_exp
                            box_h = anchors[pred][1] * h_exp
                            box_x = center_x-(box_w/2.0) 
                            box_y = center_y-(box_h/2.0) 
                          
                            classids.append(mxi)
                            confidences.append(float(mx))
                            boxes.append([box_x, box_y, box_w, box_h])
                            
                            
                            #print("box: tx:{} ,ty:{} ,tw:{} ,th:{}, center_x:{}, center_y:{}, box_w:{}, box_h:{}".format(tx,ty,tw,th,center_x,center_y, box_w, box_h))
                            
    
    #non maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold_conf, threshold_nms)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        result.append({"conf":confidences[i], "class-idx":classids[i], "box_top_x":left, "box_top_y":top, "box_lower_x":left+width, "box_lower_y":top+height})

   
    return result


def inference(frame):
    stats = {}
    inference_start = current_milli_time()

    original = frame.copy()

    inpWidth = frame.shape[1]
    inpHeight = frame.shape[0]


    n = 1
    c = 3
    h = 416
    w = 416


    print(frame.shape)
    try:
        frame = cv.resize(frame, (w,h))
    except Exception as e:
        print(str(e))


    frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW                                                      
    frame = frame.reshape((n, c, h, w))                             

    # Sets the input to the network
    net.setInput(frame)

    # Runs the forward pass to get output of the output layers
    t = current_milli_time()
    outs = net.forward(getOutputsNames(net))
    stats["forward-pass-ms"] = current_milli_time() - t;

    # process the result
    t = current_milli_time()
    result = processouts(frame, outs)
    stats["postprocess-ms"] = current_milli_time() - t;

    xscale = inpWidth / w
    yscale = inpHeight / h

    exh = 10
    fontscale = 0.5
    fontthickness=1
    font = cv.FONT_HERSHEY_SIMPLEX
    class_info = []
    for r in result:
        box_top_x = r["box_top_x"]
        box_top_y = r["box_top_y"]
        box_bottom_x = r["box_lower_x"]
        box_bottom_y = r["box_lower_y"]
        classid = r["class-idx"]
        confidence = r["conf"]
        
        #bounding box
        cv.rectangle(original, (int(box_top_x*xscale),int(box_top_y*yscale)), (int(box_bottom_x*xscale),int(box_bottom_y*yscale)), (255,70,70), 2)

        #label
        text = classes[classid] + " : " + format(confidence, '.2f')
        text_size = cv.getTextSize(text, font, fontscale, fontthickness)
        tx = int(box_top_x*xscale)
        ty = int(box_top_y*yscale) - (text_size[0][1] + exh)
        if ty < 0:
            ty = 0
        if tx < 0:
            tx = 0
        tx2 = tx + text_size[0][0]
        ty2 = ty + text_size[0][1] + exh
        if tx2 < inpWidth and ty2 < inpHeight:
            org = (int(box_top_x*xscale),int(box_top_y*yscale))
            cv.rectangle(original, (tx,ty),(tx2,ty2),(255,70,70), cv.FILLED)
            cv.putText(original, text, (tx,ty+text_size[0][1] + int((exh/2))), font, fontscale, (255,255,255), fontthickness, cv.LINE_AA) 
        
        class_info.append([classid, classes[classid], confidence])

    stats["total-ms"] = current_milli_time() - inference_start
    stats["class-info"] = class_info

    return stats, original


def load_model():
    global net
    global classes
    net = cv.dnn_DetectionModel("frozen_darknet_yolov3_model.xml",
                                "frozen_darknet_yolov3_model.bin")
    net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

    classesFile = "coco.names";
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')


## Web API ##
img_cache = []
@app.route('/inference', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['inference-file']
        #https://medium.com/csmadeeasy/send-and-receive-images-in-flask-in-memory-solution-21e0319dcc1
        npimg = np.frombuffer(f.read(), dtype=np.uint8)
        frame = cv.imdecode(npimg,cv.IMREAD_COLOR)
        
        fname = secure_filename(f.filename)
        cache_key = str(uuid.uuid1())
        
        info = {"auto_submit": 1,
                "rescale": 0}

        #rescale is only for consistent UI
        if request.form.get("rescale") is not None:
            w = frame.shape[1]
            h = frame.shape[0]
            frame = cv.resize(frame, (800, int(h*(800/w))))
            info["rescale"] = 1

        stats, result = inference(frame)
        class_info = stats["class-info"]
        print("stats", stats)

        if len(img_cache) >= 3:
            img_cache.pop()
        
        if request.form.get("autosubmit") is None:
            info["auto_submit"] = 0

        #create jpeg first
        is_success, jpg = cv.imencode(".jpg", result)
        if is_success:
            img_cache.insert(0, [cache_key, jpg])
            return render_template('result.html', image_res=cache_key, image_src=fname, stats=stats, info=info)
    
    info = {"auto_submit": 0, "rescale": 1}
    return render_template('result.html', image_res=None, auto_submit=0, info=info)



@app.route("/imgs/<path:imgpath>")
def images(imgpath):
    fullpath = imgpath
    for e in img_cache:
        print (e[0])
        if e[0] == fullpath:
            image = e[1]
            return send_file(io.BytesIO(image),
                     attachment_filename=imgpath,
                     mimetype='image/jpg')

    image = open("statics/notfound.png", "rb").read()
    return send_file(io.BytesIO(image),
                     attachment_filename="notfound.png",
                     mimetype='image/png')


@app.route('/statics/<path:path>')
def send_statics(path):
    return send_from_directory('statics', path)

@app.route('/')
def index():
    info = {"auto_submit": 0, "rescale": 1}

    return render_template('result.html', image_res=None, info=info)


if __name__ == "__main__":
    load_model()
    app.run(host= '0.0.0.0')

