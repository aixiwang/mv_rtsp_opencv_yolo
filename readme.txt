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


[Preparation]
*Win7 64bit (you can try linux by yourself)
*Winpython64-3.7.6.0Ps2
*opencv_python-4.1.2+contrib-cp37-cp37m-win_amd64.whl
*download yolov3.weights & yolov3-tiny.weights to current folder

[Camera]
1920*1080, RTSP camera
RTSP URI: rtsp://192.168.2.250:554/user=admin&password=&channel=1&stream=0.sdp?
Please check your own url to test.

[Run]
python detect.py detect_method

supported detect_method:


