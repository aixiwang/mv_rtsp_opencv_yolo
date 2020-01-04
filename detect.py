# -*- coding: utf-8 -*-
#!/usr/bin/env python
#--------------------------------------------------------------------------
# DETECT
# Copyright by Aixi Wang <aixi.wang@hotmail.com>
#
#--------------------------------------------------------------------------
# [v1 2020.1.4]
# *Created basic framework
#
# TODO: 
# *integration with GUI
#
#--------------------------------------------------------------------------
import cv2
import cv2
import numpy as np
import os
import time
import sys
import _thread
   
if len(sys.argv) > 1:
    input_detect_method = (sys.argv[1])
else:
    input_detect_method = 'cv'
        
#--------------------------------------------------------------------------
# global setting
#--------------------------------------------------------------------------
videoStreamAddress = "rtsp://192.168.2.250:554/user=admin&password=&channel=1&stream=0.sdp?"

DEBUG_EN = 0

t1 = time.time()
if input_detect_method == 'yolov3-tiny':
    print('Load yolov3-tiny model......')
    YOLO_CONFIG_PATH =  'yolov3-tiny.cfg'
    YOLO_WEIGHTS_PATH = 'yolov3-tiny.weights'
    YOLO_LABLE_PATH = 'coco.names'

elif input_detect_method == 'yolov3':
    print('Load yolov3 model......')
    YOLO_CONFIG_PATH =  'yolov3.cfg'
    YOLO_WEIGHTS_PATH = 'yolov3.weights'
    YOLO_LABLE_PATH = 'coco.names'

elif input_detect_method == 'cv':
    pass
    
else:
    print('unsupport detect_method, no init global setting,exit')
    sys.exit(-1)
    
#--------------------------------------------------------------------------
# global variables
#-------------------------------------------------------------------------- 
cap = None
cap_in_flag = 0  
img_processing_flag = 0
new_ret = 0
new_frame = None

#--------------------------------------------------------------------------
# init
#-------------------------------------------------------------------------- 
if input_detect_method == 'yolov3' or input_detect_method == 'yolov3-tiny':
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
    # 加载类别标签文件
    LABELS = open(YOLO_LABLE_PATH).read().strip().split("\n")
    nclass = len(LABELS)

    t2 = time.time()
    print('load model time(sec):',t2-t1)  



#--------------------------------------------------------------------------
# functions
#-------------------------------------------------------------------------- 

#-------------------------
# yolo_detect_py
#-------------------------
def yolo_detect_py(imgIn,
                 imgOut,
                confidence_thre=0.5,
                nms_thre=0.3,
                jpg_quality=80):

    '''
    imgIn：原始图片
    imgOut：结果图片
    confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
    nms_thre：非极大值抑制的阈值，默认为0.3
    jpg_quality：设定输出图片的质量，范围为0到100，默认为80，越大质量越好
    '''
    global cap
    global cap_in_flag
    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')

    # 载入图片并获取其维度
    #base_path = os.path.basename(pathIn)
    #img = cv2.imread(pathIn)
    #(H, W) = img.shape[:2]
    (H, W) = imgIn.shape[:2]

    # 加载模型配置和权重文件

    # 获取YOLO输出层的名字
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 将图片构建成一个blob，设置图片尺寸，然后执行一次
    # YOLO前馈网络计算，最终获取边界框和相应概率
    blob = cv2.dnn.blobFromImage(imgIn, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # 显示预测所花费时间
    print('yolo detect time(sec):',(end - start))

    # 初始化边界框，置信度（概率）以及类别
    boxes = []
    confidences = []
    classIDs = []

    # 迭代每个输出层，总共三个
    for output in layerOutputs:
        # 迭代每个检测
        for detection in output:
            # 提取类别ID和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # 只保留置信度大于某值的边界框
            if confidence > confidence_thre:
                # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是
                # 边界框的中心坐标以及边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")


                # 计算边界框的左上角位置
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # 更新边界框，置信度（概率）以及类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)

    # 确保至少一个边界框
    if len(idxs) > 0:
        # 迭代每个边界框
        for i in idxs.flatten():
            # 提取边界框的坐标
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 绘制边界框以及在左上角添加类别标签和置信度
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(imgOut, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(imgOut, (x, y-text_h-baseline), (x + text_w, y), color, -1)
            cv2.putText(imgOut, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 输出结果图片
    #if pathOut is None:
    #    cv2.imwrite('with_box_'+base_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    #else:
    #    cv2.imwrite(pathOut, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
  
#-------------------------
# loop_read_cam
#-------------------------   
def loop_read_cam():
    global cap
    global cap_in_flag
    global img_processing_flag
    global new_ret
    global new_frame

    while True:
        if cap.isOpened():
            if img_processing_flag == 1:
                
                pass
            else:
                cap_in_flag = 1
                
                new_ret, new_frame = cap.read()
                
                cap_in_flag = 0
        else:
            #print('cap is non opened')
            pass
                        
#-------------------------
# run_camera
#-------------------------   
def run_camera(w,h,detect_method= 'cv'):

    global cap
    global cap_in_flag
    global img_processing_flag
    global new_ret
    global new_frame
    
    
    print('1. open camera')
    # Create a VideoCapture object and read from RTSP protocol
    cap = cv2.VideoCapture(videoStreamAddress)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    
    print('2. start reading cam thread')
    _thread.start_new_thread(loop_read_cam,())
    #time.sleep(1)
    
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    
    
    th = 100
    
    
    if detect_method == 'cv':
        cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    # Read until video is completed

    
    print('3. start image processing loop')
    
    index = 0
    while True:
        t1 = time.time()
        # Capture frame-by-frame
        #ret = True
        #while ret == True:
        #    ret, frame = cap.read()
        #frame = None
        #ret = False
        #ret, frame = cap.read()
        
        
        if new_ret == True:
            index += 1
            #cv2.imwrite('raw.jpg', new_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            #new_frame_co = cv2.resize(new_frame,(0,0),fx=0.5,fy=0.5)
            new_frame_co = new_frame.copy()
            
            if detect_method == 'cv':
                gray = cv2.cvtColor(new_frame_co, cv2.COLOR_RGB2GRAY)
                gray[gray <= th] = 1
                gray[gray > th] = 0
                gray[gray == 1] = 255
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in contours:
                    (x, y), r = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    r = int(r)
                    cv2.circle(new_frame_co, center, 5, (0, 255, 255), 8)
                    cv2.circle(new_frame_co, center, r, (0, 255, 255), 5)
                # Display the resulting frame
                cv2.imshow('gray', gray)
            
            elif detect_method == 'yolov3-tiny' or detect_method == 'yolov3':
                yolo_detect_py(new_frame_co,new_frame_co)
                cv2.imshow('main', new_frame_co)
                if DEBUG_EN == 1:
                    cv2.imwrite('debug_output\detected' + str(index) + '.jpg', new_frame_co, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                # Press Q on keyboard to  exit
            
            else:
                print('unsupported detected method')
                pass


            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            
            t2 = time.time()
            print('one loop time(sec):',t2-t1)            
        # Break the loop
        #else:
        #    #break
            
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

#-------------------------
# run_camera
#-------------------------
if __name__ == "__main__":
    print('Usage: python detect.py detect_method')
    print('       python detect.py cv/yolov3/yolov3-tiny')

    
    run_camera(1920,1080,detect_method = input_detect_method)