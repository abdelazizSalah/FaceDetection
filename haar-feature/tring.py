# importing the libraries
import cv2
import numpy as np

# detecting the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Setup camera
cap = cv2.VideoCapture(0)

# Read logo and resize
path = './filters/hats/'
logo = cv2.imread(path+'chicken.png')


while cap.isOpened():
    # Capture frame-by-frame
    ret, img = cap.read()

    # detecting the face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    upOrDown = 1
    # Region of Image (ROI), where we want to insert logo
    # roi = img[-size-10:-10, -size-10:-10]
    # it takes the positions where you want to put the logo, but it must be the same size of the logo
    for (leftPosX, topLeftY, width, height) in faces:
        cv2.rectangle(img, (leftPosX, topLeftY), (leftPosX +
                      width, topLeftY+height), (0, 255, 0), 3)
        size = width - 50

        # print(leftPosX, y, width, h)
        logo = cv2.resize(logo, (size, size))

        # Create a mask of logo
        img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        # roi = img[topLeftY - size:topLeftY, leftPosX:leftPosX + size]
        roi = img[topLeftY - size:topLeftY, leftPosX:leftPosX + size]

        # Set an index of where the mask is
        roi[np.where(mask)] = 0
        roi += logo

        cv2.imshow('WebCam', img)
        if cv2.waitKey(1) == ord('q'):
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        elif cv2.waitKey(1) == ord('a'):
            print('clown')
            logo = cv2.imread(path+'clown-hat.png')
        elif cv2.waitKey(1) == ord('s'):
            logo = cv2.imread(path+'birthday-hat.png')
        elif cv2.waitKey(1) == ord('d'):
            logo = cv2.imread(path+'complete-clown-hat.png')
        elif cv2.waitKey(1) == ord('f'):
            logo = cv2.imread(path+'crying-frog-hat.png')
        elif cv2.waitKey(1) == ord('g'):
            logo = cv2.imread(path+'kings-crown.png')
        elif cv2.waitKey(1) == ord('h'):
            logo = cv2.imread(path+'pig-hat.png')
        elif cv2.waitKey(1) == ord('j'):
            logo = cv2.imread(path+'wizzard-hat.png')
