import cv2 as cv

haarcascade = "haarcascade_frontalface_default.xml"

cap = cv.VideoCapture(0)

cap.set(3,640) #widht
cap.set(4,480) #height

while True: 
    success, img = cap.read()

    facecascade = cv.CascadeClassifier(haarcascade)
    img_gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    face = facecascade.detectMultiScale(img_gray, 1.1,4)

    for (x,y,w,h) in face:
     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

    cv.imshow("Face", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break