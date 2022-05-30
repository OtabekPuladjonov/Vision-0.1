import cv2

cap = cv2.VideoCapture(0)

count = 0
l = []
while True:

    ret, frame = cap.read()
    count += 1
    l.append(frame)
    if count > 50:
        cv2.imwrite("NewPicture10.jpg", l[1])
        cv2.imwrite("NewPicture1.jpg", l[50])
        l.remove(l[0])

        cv2.destroyAllWindows()

