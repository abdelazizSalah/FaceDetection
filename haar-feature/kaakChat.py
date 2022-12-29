# importing the libraries
import cv2
import numpy as np

# detecting the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Setup camera
cap = cv2.VideoCapture(0)


# Read logo and resize
path = './filters/'
logo = cv2.imread(path+'hats/rabbit-ears.png')
fromY = 0
toY = 0
up = 1
imgToBeShown = 0
stillImg = True

while cap.isOpened():
    # Capture frame-by-frame
    ret, img = cap.read()

    # detecting the face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if(stillImg == False):
    #     img = gray

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

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

        if(up == 1):
            fromY = topLeftY - size
            toY = topLeftY
        else:
            fromY = topLeftY + height
            toY = topLeftY + height + size

        roi = img[fromY:toY, leftPosX:leftPosX + size]
        # roi = img[leftPosX:leftPosX + size, fromY:toY]

        # Set an index of where the mask is
        roi[np.where(mask)] = 0
        roi += logo

        cv2.imshow('WebCam', gray)
        if cv2.waitKey(1) == ord('q'):
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        elif cv2.waitKey(1) == ord('a'):
            print('clown')
            logo = cv2.imread(path+'hats/clown-hat.png')
            up = 1
        elif cv2.waitKey(1) == ord('s'):
            logo = cv2.imread(path+'hats/birthday-hat.png')
            up = 1
        elif cv2.waitKey(1) == ord('d'):
            up = 1
            logo = cv2.imread(path+'hats/complete-clown-hat.png')
        elif cv2.waitKey(1) == ord('f'):
            up = 1
            logo = cv2.imread(path+'hats/crying-frog-hat.png')
        elif cv2.waitKey(1) == ord('g'):
            up = 1
            logo = cv2.imread(path+'hats/kings-crown.png')
        elif cv2.waitKey(1) == ord('h'):
            up = 1
            logo = cv2.imread(path+'hats/pig-hat.png')
        elif cv2.waitKey(1) == ord('j'):
            up = 1
            logo = cv2.imread(path+'hats/wizzard-hat.png')
        elif cv2.waitKey(1) == ord('z'):
            up = 0
            logo = cv2.imread(path+'beards/long-beard.png')
        elif cv2.waitKey(1) == ord('x'):
            up = 0
            logo = cv2.imread(path+'beards/m2ashaBeard.png')
        elif cv2.waitKey(1) == ord('c'):
            up = 0
            logo = cv2.imread(path+'beards/santa-beard.png')

        elif cv2.waitKey(1) == ord('w'):
            imgToBeShown = gray
            stillImg = False
        elif cv2.waitKey(1) == ord('e'):
            up = 0
            logo = cv2.imread(path+'beards/santa-beard.png')
        elif cv2.waitKey(1) == ord('r'):
            up = 0
            logo = cv2.imread(path+'beards/santa-beard.png')
