import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color= (180,180,180))

#HOLISTIC MODEL
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance
    image.flags.writeable = False

    # Make detections
    results = holistic.process(image)

    # To improve performance
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image = image,
                    landmark_list = results.face_landmarks,
                    connections = mp_holistic.FACEMESH_CONTOURS # FACEMESH_TESSELATION, FACEMESH_CONTOURS
                    ,landmark_drawing_spec = drawing_spec ,connection_drawing_spec = drawing_spec
                    )

    mp_drawing.draw_landmarks(image = image,
                    landmark_list = results.right_hand_landmarks,
                    connections = mp_holistic.HAND_CONNECTIONS 
                    ,landmark_drawing_spec = drawing_spec ,connection_drawing_spec = drawing_spec
                    )

    mp_drawing.draw_landmarks(image = image,
                    landmark_list = results.left_hand_landmarks,
                    connections = mp_holistic.HAND_CONNECTIONS
                    ,landmark_drawing_spec = drawing_spec ,connection_drawing_spec = drawing_spec
                    )

    mp_drawing.draw_landmarks(image = image,
                    landmark_list = results.pose_landmarks,
                    connections = mp_holistic.POSE_CONNECTIONS 
                    #,landmark_drawing_spec = drawing_spec ,connection_drawing_spec = drawing_spec
                    )



    image=cv2.flip(image, 1)
    fps = 1 / (time.time() - start)
    cv2.putText(image, f"FPS: {int (fps)}", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow("Raw webcam feed",image)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break



cap.release()
cv2.destroyAllWindows()