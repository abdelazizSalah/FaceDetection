import cv2

# reading the img
img = cv2.imread('TrialImg.jpg')

# converting the img into grey scale
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# loading the haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# detecting the faces, we should try many scales till we reach the best one.
detected_faces = face_cascade.detectMultiScale(img, 1.3, 4)

# drawing the rectangle around the detected faces
for (x, y, w, h) in detected_faces:
    # lets draw green rectangles.
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# showing the img
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
