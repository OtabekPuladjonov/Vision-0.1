import cv2
import numpy as np


thres = 0.6
#Threshold to detect object
nms_threshold = 0.2
lang = 'en'

img = cv2.imread("NewPicture.jpg")


classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

name_count=set()
while True:
    #success,img = img.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    cv2.imshow("Output",img)

    if cv2.waitKey(1) & 0xFF==ord('q'):


        break