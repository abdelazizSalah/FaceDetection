# @author Abdelaziz Salah
# @date 31/12/2022
# @breif This code is a simple implementation of animated snapchat filters using opencv and mediapipe

# importing the libraries
import cv2
import numpy as np
import itertools
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# detecting the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# lets initialize the mediapipe face detection model
mp_face_detection = mp.solutions.face_detection

# lest setup the face detection model
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5, model_selection=0)

# initialize the mediapipe face drawing model
mp_drawing = mp.solutions.drawing_utils


# now lets work with media pipe
mp_face_mesh = mp.solutions.face_mesh

# lets setup the face mesh model for static photos
face_mesh_images = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=6, min_detection_confidence=0.5)

# lets setup the face mesh model for videos
face_mesh_video = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=6, min_detection_confidence=0.5)

# now we initialize the mediapipe drawing styles
mp_drawing_styles = mp.solutions.drawing_styles

# now we can create a face landmarks detection function


def detectFacialLandmarks(image, face_mesh, display=True):
    '''
        This function performs facial landmarks detection on an image and displays it.
        ARGS:
            image: the image to perform the detection on
            face_mesh: the face landmarks detection function required to perform the landmarks detection.
            display: boolean value to detect whether to display the image or not

        Returns:
            output_image: the image with the landmarks drawn on it.
            results: the output of thefacial landmarks detection on the input image
    '''

    # Perform the facial landmarks detection on the image , after converting it into RGB
    results = face_mesh.process(image[:, :, ::-1])

    # create a copy from the image
    output_image = image[:, :, ::-1].copy()

    # Check if the facial landmarks exists
    if results.multi_face_landmarks:

        # loop over all the faces in the image
        for face_landmarks in results.multi_face_landmarks:

            # draw the facial landmarks on the output image with the face mesh tesselation
            mp_drawing.draw_landmarks(  # da byrsm el khtot el 3l wesh kolo
                image=output_image,
                landmark_list=face_landmarks,
                # we can change this to only draw the eyes or nose or mouth,etc...
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # draw the facial landmarks on the output image with the countours
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # check if we need to display the image
    if display:
        # specify the image size
        plt.figure(figsize=(10, 10))

        # Display the original image
        plt.subplot(121)
        plt.title('Original Image')
        plt.axis('off')
        plt.imshow(image[:, :, ::-1])

        # Display the image
        plt.subplot(122)
        plt.title('Output image')
        plt.axis('off')
        plt.imshow(output_image)
        plt.show()

    # otherwise
    else:  # this case we need it in videos because we just need the data without ploting static images.
        # return the output image and the results
        return np.ascontiguousarray(output_image[:, :, ::-1], dtype=np.uint8), results


# now we need to make a function that get the size of face parts and from them we can detect face expressions.
def getSize(image, face_landmarks, INDCIES):
    '''
        this function is used to calculate the width and height of the face parts.
        ARGS:
            image: the image to perform the detection on
            face_landmarks: the landmarks of the face
            INDCIES: the indices of the face parts to calculate the size of
        Returns:
            width: the width of the face part
            height: the height of the face part
            landmarks: the landmarks of the face part whose size is calculated
    '''

    # Retreive the width and height of the image
    image_height, image_width, _ = image.shape

    # convert the indcies to a list
    INDCIES_LIST = list(itertools.chain(*INDCIES))

    # initialize a list to carry the landmarks
    landmarks = []

    # iterate over the indices of the landmarks
    for index in INDCIES_LIST:
        # append the landmark to the list
        landmarks.append([int(face_landmarks.landmark[index].x * image_width),  # we multiply the x and y by the width and height of the image to get the actual coordinates of the landmark
                         int(face_landmarks.landmark[index].y * image_height)])

    # calculate the width and height of the face part
    # this function returns x, y, width, height but we won't use x,y so we can simply ignore them by using _
    _, _, width, height = cv2.boundingRect(np.array(landmarks))

    # convert the list of the width and height to a numpy array
    landmarks = np.array(landmarks)

    # return the width and height and the landmarks
    return width, height, landmarks


# now we can check whether the eyes of mouth are open or closed by calculating the size of the face parts.
def isOpen(image, face_mesh_results, face_part, threshold=5, display=True):
    '''
        this function is used to check if the face part is open or closed.
        ARGS:
            image: the image to perform the detection on
            face_mesh_results: the output of the facial landmarks detection on the image
            face_part: the face part to check if it's open or closed
            threshold: the threshold value used to chech the isOpen condition
            display: bool value that if True we display an image and returns nothing but if false we returns
                     the output image and the status
        Returns:
            output_image: the output image with status written on it.
            status: the status of the face part (open or closed),
                    which is a dictionary for all persons in the image with the status of the face part for each person
    '''

    # Retreive the width and height of the image
    image_height, image_width, _ = image.shape

    # create a copy of the image to draw on it
    output_image = image.copy()

    # Create a dictionary to store the status of the face part for each person
    status = {}

    # Check if the face part is mouth
    if face_part == 'MOUTH':
        # get the indcies of the mouth
        INDCIES = mp_face_mesh.FACEMESH_LIPS

        # specifiy the location to write the is mouse open or closed
        loc = (10, image_height - image_height // 40)

        # initialize a increment that will be added to the status writing location so that  the status of each person will be written in a different line
        increment = - 30

    elif face_part == 'LEFT EYE':
        # Get the indices of the left eye
        INDCIES = mp_face_mesh.FACEMESH_LEFT_EYE
        loc = (10, 30)
        increment = 30
    elif face_part == 'RIGHT EYE':
        # Get the indices of the right eye
        INDCIES = mp_face_mesh.FACEMESH_RIGHT_EYE
        loc = (image_width - 300, 60)
        increment = 30
    else:
        return

    # iterate over the face landmarks
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        # get the size of the face part
        _, height, _ = getSize(  # this function returns width, height, landmarks but we don't need width & landmarks so we can simply ignore it by using _
            image, face_landmarks, INDCIES)

        # get the height of the face
        _, face_height, _ = getSize(
            image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)

        # check if the face part is open
        if (height / face_height) * 100 > threshold:
            # update the status of the face part for this person
            status[face_no] = 'Open'

            # set the color to green
            color = (0, 255, 0)

        else:
            # update the status of the face part for this person
            status[face_no] = 'Closed'

            # set the color to red
            color = (0, 0, 255)

        # write the status of the face part for this person
        cv2.putText(output_image, f'Face{face_no + 1} {face_part} {status[face_no]}.', (
            loc[0], loc[1] + face_no * increment), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # check if we need to display the image
    if display:
        # Display the image
        plt.figure(figsize=(5, 5))
        plt.imshow(output_image[:, :, ::-1])
        plt.title('Output image')
        plt.axis('off')
        plt.show()
    else:
        # return the output image and the status this will be used for the video.
        return output_image, status


# now we can define a function which takes a filter over a given image
def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    '''
    This function will overlay a filter image over a face part of a person in the image/frame.
    Args:
        image:          The image of a person on which the filter image will be overlayed.
        filter_img:     The filter image that is needed to be overlayed on the image of the person.
        face_landmarks: The facial landmarks of the person in the image.
        face_part:      The name of the face part on which the filter image will be overlayed.
        INDEXES:        The indexes of landmarks of the face part.
        display:        A boolean value that is if set to true the function displays 
                        the annotated image and returns nothing.
    Returns:
        annotated_image: The image with the overlayed filter on the top of the specified face part.
    '''

    # Create a copy of the image to overlay filter image on.
    annotated_image = image.copy()

    # Errors can come when it resizes the filter image to a too small or a too large size .
    # So use a try block to avoid application crashing.
    try:

        # Get the width and height of filter image.
        filter_img_height, filter_img_width, _ = filter_img.shape

        # Get the height of the face part on which we will overlay the filter image.
        _, face_part_height, landmarks = getSize(
            image, face_landmarks, INDEXES)

        # Specify the height to which the filter image is required to be resized.
        required_height = int(face_part_height*2.5)

        # Resize the filter image to the required height, while keeping the aspect ratio constant.
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width *
                                                         (required_height/filter_img_height)),
                                                     required_height))

        # Get the new width and height of filter image.
        filter_img_height, filter_img_width, _ = resized_filter_img.shape

        # Convert the image to grayscale and apply the threshold to get the mask image.
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)

        # Calculate the center of the face part.
        center = landmarks.mean(axis=0).astype("int")

        # Check if the face part is mouth.
        if face_part == 'MOUTH':

            # Calculate the location where the smoke filter will be placed.
            location = (int(center[0] - filter_img_width / 3), int(center[1]))

        # Otherwise if the face part is an eye.
        elif face_part == 'EYE':

            # Calculate the location where the eye filter image will be placed.
            location = (int(center[0]-filter_img_width/2),
                        int(center[1]-filter_img_height/2))

        elif face_part == 'Oval':

            # Calculate the location where the eye filter image will be placed.
            location = (int(center[0]-filter_img_width/2),
                        int(center[1]-filter_img_height/2))

        # Retrieve the region of interest from the image where the filter image will be placed.
        ROI = image[location[1]: location[1] + filter_img_height,
                    location[0]: location[0] + filter_img_width]

        # Perform Bitwise-AND operation. This will set the pixel values of the region where,
        # filter image will be placed to zero.
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)

        # Add the resultant image and the resized filter image.
        # This will update the pixel values of the resultant image at the indexes where
        # pixel values are zero, to the pixel values of the filter image.
        resultant_image = cv2.add(resultant_image, resized_filter_img)

        # Update the image's region of interest with resultant image.
        annotated_image[location[1]: location[1] + filter_img_height,
                        location[0]: location[0] + filter_img_width] = resultant_image

    # Catch and handle the error(s).
    except Exception as e:
        pass

    # Check if the annotated image is specified to be displayed.
    if display:

        # Display the annotated image.
        plt.figure(figsize=[10, 10])
        plt.imshow(annotated_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the annotated image.
        return annotated_image


# Read the filter hat or beard and resize
path = './filters/'
logo = cv2.imread(path+'face_land_marks_filters/mouth.png')
# read the left and right eyes
left_eye = cv2.imread(path + 'face_land_marks_filters/Eye.png')
right_eye = cv2.imread(path + 'face_land_marks_filters/Eye.png')
# read the mouth filter
mouth = cv2.imread(path + 'face_land_marks_filters/mouth.png')

fromY = 0
toY = 0
up = 1
imgToBeShown = 0
stillImg = True


# Inizializ the video capture
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)
while camera_video.isOpened():
    # Capture frame-by-frame
    ok, img = camera_video.read()

    # if the captured frame got an error or wrong frame lets continue to the next frame
    if not ok:
        continue

    # detecting the face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if(stillImg == False):
    #     img = gray

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Region of Image (ROI), where we want to insert logo
    # roi = img[-size-10:-10, -size-10:-10]
    # it takes the positions where you want to put the logo, but it must be the same size of the logo
    for (leftPosX, topLeftY, width, height) in faces:
        # cv2.rectangle(img, (leftPosX, topLeftY), (leftPosX +
        #               width, topLeftY+height), (0, 255, 0), 3)

        size = width

        # print(leftPosX, y, width, h)
        logo = cv2.resize(logo, (size, size))

        # Create a mask of logo
        img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

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

        # showing the image.
        cv2.imshow('KaakChaat', img)

        # changing the filter depending on the key pressed.
        if cv2.waitKey(1) == ord('q'):
            # When everything done, release the camera_videoture
            camera_video.release()
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
            logo = cv2.imread(path+'hats/Lion-hat.png')
        elif cv2.waitKey(1) == ord('k'):
            up = 1
            logo = cv2.imread(path+'hats/santa_2.png')

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
