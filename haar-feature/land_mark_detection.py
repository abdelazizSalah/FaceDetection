import cv2
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#         static_image_mode=True,
#         max_num_faces=1,
#         min_detection_confidence=0.5) as face_mesh:
#     image = cv2.imread('Khaled.jpg')
#     # Convert the BGR image to RGB before processing.
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if not results.multi_face_landmarks:
#         print('continue')
#     annotated_image = image.copy()
#     for face_landmarks in results.multi_face_landmarks:
#         print('face_landmarks:', face_landmarks)
#         print(
#             f'index 0: {face_landmarks.landmark[0]}'
#         )
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=drawing_spec,
#             connection_drawing_spec=drawing_spec)
#     cv2.imwrite('/tmp/annotated_image.png', annotated_image)

# # For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
cap = cv2.VideoCapture(0)
prev_Time = 0
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
    # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        curr_Time = time.time()
        fps = 1/(curr_Time - prev_Time)
        prev_Time = curr_Time
        cv2.putText(image, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
