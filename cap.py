import cv2

videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    succes,image = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg",image)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()