import cv2 as cv

img = cv.imread('Photos/group 1.jpg')
#cv.imshow('person',img)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray Person', gray_img)

haar_cascade = cv.CascadeClassifier('Haarcascade_frontalface_default.xml') 

faces_rect = haar_cascade.detectMultiScale( gray_img, scaleFactor = 1.1, minNeighbors = 3) 

print(f'number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect: 
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 

cv.imshow('Detected faces', img) 


cv.waitKey(0)

'''import cv2 as cv
cap = cv.VideoCapture(0)
while True: 
    
    ret,img=cap.read()
    
    cv.imshow('Video', img)
    
    if(cv.waitKey(10) & 0xFF == ord('b')):
        break'''
