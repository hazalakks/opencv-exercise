import cv2 as cv

face_cascade_name = 'haarcascade_frontalface_default.xml'
eyes_cascade_name = 'haarcascade_eye_tree_eyeglasses.xml'

# Load the cascade classifiers
face_cascade = cv.CascadeClassifier(cv.samples.findFile(face_cascade_name))
eyes_cascade = cv.CascadeClassifier(cv.samples.findFile(eyes_cascade_name))

# Open the video capture device (default camera)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        
        faceROI = frame_gray[y:y + h, x:x + w]
       
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    
    # Display the frame with detections
    cv.imshow('Face detection', frame)

    if cv.waitKey(10) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
