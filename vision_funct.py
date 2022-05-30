import cv2
#get input from main camera
img = cv2.imread("D:/MACHINE LEARNING/COMPUTER VISION/hand tracking/NewPicture.jpg")
#set the size and brightness



thres = 0.60 # Threshold to detect object
classNames= []
#object names of trained data
classFile = 'coco.names'
#reading object names one by one
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#dataset values
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#setting up cv2 dnn detection model for getting values from dataset and input scaling
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#set for collecting unique object names
name_count=set()
#reading the image and detecting the objects in it
while True:

    classIds, confs, bbox = net.detect(img,confThreshold=thres)
#reading object names found in original name from id numbers and replacing them into a list
    name=[]
    if len(classIds) != 0:
        for classId in (classIds.flatten()):
            cln = classNames[classId - 1]

            name.append(cln)
        #take off the unique values from the object found list
        name_count.update(name)
        print(name_count)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

"""
#display main cam video
    cv2.imshow("Output", img)
#break if q is pressed
"""
