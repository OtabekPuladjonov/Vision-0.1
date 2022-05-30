import cv2
import mediapipe as mp
import pyttsx3
import time
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import urllib
import os

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,320)

detector=HandDetector(detectionCon=0.8, maxHands=1)

x= [400,300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y=[15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

count = 0
l = []

coff = np.polyfit(x,y,2) #y=Ax^2=Bx+C
lmlist=[]
while True:
    success,img =cap.read()
    hands,img =detector.findHands(img)
    distCm=300
    count += 1
    l.append(img)
    if count > 60:
        cv2.imwrite("NewPicture10.jpg", l[1])
        cv2.imwrite("NewPicture1.jpg", l[60])
        l.remove(l[0])
    if hands:
        lmlist = hands[0]['lmList']
        x1, a, a1 = lmlist[5]
        x2, b, b2 = lmlist[17]

        dist = int(round(math.sqrt((b - a) ** 2 + (x2 - x1) ** 2)))
        A, B, C = coff

        distCm = A * dist ** 2 + B * dist + C
        distCm = int(round(distCm))

    if distCm<=150:

        #cv2.imwrite("NewPicture10.jpg", img)
        capture = cv2.imread("NewPicture10.jpg")
        #capture = f.image
        thres = 0.6  # Threshold to detect object
        classNames = []
        # object names of trained data
        classFile = 'coco.names'
        # reading object names one by one
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        # dataset values
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        # setting up cv2 dnn detection model for getting values from dataset and input scaling
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        # set for collecting unique object names
        name_count = set()
        # reading the image and detecting the objects in it
        result = True
        while (result):
            classIds, confs, bbox = net.detect(capture, confThreshold=thres)

            # reading object names found in original name from id numbers and replacing them into a list
            name = []
            if len(classIds) != 0:

                for classId in (classIds.flatten()):
                    cln = classNames[classId - 1]
                    #print(cln)
                    name.append(cln)
                # take off the unique values from the object found list
                name_count.update(name)
                #print the name of the objects found
                print(f"detected object: {name_count}")
                result=False
                #read out the names of object
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                n_s = list(name_count)
                sound=engine.say("you're facing to")

                for i in n_s:
                    var = i
                    engine.say(var)
                engine.runAndWait()
                os.remove('NewPicture10.jpg')
                os.remove('NewPicture1.jpg')

    #show output video

    cv2.imshow('live video', img)
    #wait till q is pressed for breaking the process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()