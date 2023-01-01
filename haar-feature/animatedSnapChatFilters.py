# @author Abdelaziz Salah
# @date 31/12/2022
# @breif This code is a simple implementation of animated snapchat filters using opencv and mediapipe


import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt


# lets initialize the mediapipe face detection model
mp_face_detection = mp.solutions.face_detection

# lest setup the face detection model
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5, model_selection=0)

# initialize the mediapipe face drawing model
mp_drawing = mp.solutions.drawing_utils

# now lets read the image
sample_image = cv2.imread('Me.jpeg')

# specify the image size
# plt.figure(figsize=(10, 10))

# Display the image and also convert BGR to RGB for display
# plt.title('MyImage')
# plt.axis('off')
# plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
# plt.show()


# perform face detection after converting the image into RGB
face_detection_results = face_detection.process(
    cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

# check if the face(s) in the image are found
if face_detection_results.detections:
    # loop over all the faces in the image
    for face_no, face in enumerate(face_detection_results.detections):
        # Display number of faces detected
        print(f'Face {face_no+1} detected')
        # diplay the confidence of the face detection
        print(f'Confidence: {face.score}')
        # get the face bounding box and face key points coordinates
        face_bounding_box = face.location_data

        # display the face bounding box
        print(f'Face bounding box: {face_bounding_box.relative_bounding_box}')

        # iterate 6 times to get the 6 key points
        for i in range(6):
            print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
            print(
                f'{face_bounding_box.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')

# now lets create a copy from the result
# this is another way to convert BGR to RGB
img_copy = sample_image[:, :, ::-1].copy()

# check if the face(s) in the image are found
if face_detection_results.detections:

    # loop over all the faces in the image
    for face_no, face in enumerate(face_detection_results.detections):
        # Draw the face bounding box and key points on the copy of the image
        mp_drawing.draw_detection(image=img_copy, detection=face, keypoint_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2),)
# specify the image size
fig = plt.figure(figsize=(10, 10))

# Display the image
# plt.title('MyImage')
# plt.axis('off')
# plt.imshow(img_copy)
# plt.show()


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


# now we need to perform face landmarks detectin on the image after converting it to RGB
face_mesh_results = face_mesh_images.process(sample_image[:, :, ::-1].copy())

# Ger the list of indcied for all landmarks
LEFT_EYE_INDICES = list(
    set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDICES = list(
    set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
MOUTH_INDICES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
# THIS IS THE WHOLE FACE INDCIES
OVAL_INDICES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
RIGHT_EAR_INDICES = list(
    set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
LEFT_EAR_INDICES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))

# now lets check that they really exists and not empty
if face_mesh_results.multi_face_landmarks:

    # loop over all the faces in the image
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

        # display the face number
        print(f'Face {face_no+1} detected')
        print('-------------------')

        # Display the name of this landmark
        print('LEFT EYE LANDMARKS\n')

        for LEFT_EYE_INDEX in LEFT_EYE_INDICES[:2]:
            print(
                f'{face_landmarks.landmark[LEFT_EYE_INDEX]}:')

        print('RIGHT EYE LANDMARKS\n')
        for RIGHT_EYE_INDEX in RIGHT_EYE_INDICES[:2]:
            print(
                f'{face_landmarks.landmark[RIGHT_EYE_INDEX]}:')
        print('MOUTH EYE LANDMARKS\n')
        for MOUTH_INDEX in MOUTH_INDICES[:2]:
            print(
                f'{face_landmarks.landmark[MOUTH_INDEX]}:')
        print('NOSE LANDMARKS\n')
        for NOSE_INDEX in OVAL_INDICES[:2]:
            print(
                f'{face_landmarks.landmark[NOSE_INDEX]}:')

        # el t7t dol bygebo errors
        #  print('LEFT EAR LANDMARKS\n')
        # for LEFT_EAR_INDEX in LEFT_EAR_INDICES[:2]:
        #     print(
        #         f'{face_landmarks.landmark[LEFT_EAR_INDEX]}:')
        # print('RIGHT EAR LANDMARKS\n')
        # for RIGHT_EAR_INDEX in RIGHT_EAR_INDICES[:2]:
        #     print(
        #         f'{face_landmarks.landmark[RIGHT_EAR_INDEX]}:')


# now we need to draw them on the image
img_copy = sample_image[:, :, ::-1].copy()

# check if the face(s) in the image are found
if face_mesh_results.multi_face_landmarks:

    # loop over all faces
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        # draw the face landmarks on the image
        mp_drawing.draw_landmarks(  # da byrsm el khtot el 3l wesh kolo
            image=img_copy,
            landmark_list=face_landmarks,
            # we can change this to only draw the eyes or nose or mouth,etc...
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        # mp_drawing.draw_landmarks(
        #     image=img_copy,
        #     landmark_list=face_landmarks,
        #     # we can change this to only draw the eyes or nose or mouth,etc...
        #     connections=mp_face_mesh.FACEMESH_LIPS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_lips_style())

        # draw the facial landmarks with the face mesh contours -> da byrsm el landmarks bs
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

# specify the image size
fig = plt.figure(figsize=(10, 10))

# Display the image
# plt.title('MyImage')
# plt.axis('off')
# plt.imshow(img_copy)
# plt.show()


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


# # now lets make sure that the function works
# image = cv2.imread('Me.jpeg')
# detectFacialLandmarks(image, face_mesh_images, True)


# image = cv2.imread('Khaled.jpg')
# detectFacialLandmarks(image, face_mesh_images, True)


# now lets try it with web cam real time video.

# Initialize the videoCapture to read from the webCam
# camera_video = cv2.VideoCapture(0)
# camera_video.set(3, 1280)
# camera_video.set(4, 960)  # setting the resolutions


# # initialize varible to store time inside it.
# time1 = 0

# while camera_video.isOpened():
#     # read a frame
#     ok, frame = camera_video.read()

#     # check f the frame is not read properly then just continue
#     if not ok:
#         continue

#     # Flip frame horizontaly that to work of the real image not the flip of it( Ambulance idea )
#     frame = cv2.flip(frame, 1)  # 1 is the flip code

#     # perform face landmark detection
#     frame, _ = detectFacialLandmarks(
#         frame, face_mesh_video, display=False)

#     # set the time for this frame to the current time
#     time2 = time()

#     # Check if the difference between the previous and this frame time > 0 to avoide division by 0
#     if(time2 - time1 > 0):
#         # calculate number of frames per second
#         frames_per_second = 1.0 / (time2 - time1)

#         # write the calculated number of frames per second on the frame
#         cv2.putText(frame, 'FPS :{}'.format(int(frames_per_second)),
#                     (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

#     # update the previous frame time to this time
#     time1 = time2

#     # display the frame
#     cv2.imshow('Face landmarks Detection', frame)

#     # wait for 1ms if a key is pressedm retreive the ASCII code of it
#     k = cv2.waitKey(1) & 0xff

#     # check if esc is pressed break and close
#     if(k == 27):
#         break

# camera_video.release()
# cv2.destroyAllWindows()


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


# now we can try this on certain images.
image = cv2.imread('Me.jpeg')
image = cv2.flip(image, 1)
_, face_mesh_results = detectFacialLandmarks(
    image, face_mesh_images, display=False)
if face_mesh_results.multi_face_landmarks:
    output_image, _ = isOpen(image, face_mesh_results,
                             'MOUTH', threshold=2, display=False)
    output_image, _ = isOpen(
        output_image, face_mesh_results, 'LEFT EYE', threshold=9, display=False)
    isOpen(output_image, face_mesh_results,
           'RIGHT EYE', threshold=4, display=True)


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
        else:

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

# now after we finished the function lets try this on the web cam


# Inizializ the video capture
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)


# create named window
cv2.namedWindow("KAAKCHAT", cv2.WINDOW_NORMAL)

# read the left and right eyes
left_eye = cv2.imread('./filters/face_land_marks_filters/Eye.png')
right_eye = cv2.imread('./filters/face_land_marks_filters/Eye.png')

# read the mouth filter
mouth = cv2.imread('./filters/face_land_marks_filters/mouth.png')

# iterate until the webcam is closed
while camera_video.isOpened():
    # read the frame
    ok, frame = camera_video.read()
    if not ok:
        continue

    # flip the frame
    frame = cv2.flip(frame, 1)

    # perform landmarks detection
    _, face_mesh_results = detectFacialLandmarks(
        frame, face_mesh_video, display=False)

    # check if we have face landmarks
    if face_mesh_results.multi_face_landmarks:
        # get the status of mouth and eyes
        _, mouth_status = isOpen(
            frame, face_mesh_results, 'MOUTH', threshold=15, display=False)

        _, left_eye_status = isOpen(
            frame, face_mesh_results, 'LEFT EYE', threshold=5, display=False)
        _, right_eye_status = isOpen(
            frame, face_mesh_results, 'RIGHT EYE', threshold=5, display=False)

        # iterate over the found faces.
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            # check if the left eye is open
            if left_eye_status[face_num] == 'Open':

                # overlay the eye image
                frame = overlay(frame, left_eye, face_landmarks,
                                'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
            # check if the left eye is open
            if right_eye_status[face_num] == 'Open':

                # overlay the eye image
                frame = overlay(frame, right_eye, face_landmarks,
                                'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)

            # check if the left eye is open
            if mouth_status[face_num] == 'Open':

                # overlay the eye image
                frame = overlay(frame, mouth, face_landmarks,
                                'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)
    cv2.imshow("KAAKCHAT", frame)

    # check if the user pressed the escape key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# release the camera
camera_video.release()
cv2.destroyAllWindows()
