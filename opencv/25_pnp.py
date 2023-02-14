import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,0))


cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()

    start = time.time()
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image. flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = [] 
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # 33    : Sol Goz
            # 263   : Sag Goz 
            # 1     : Burun
            # 61    : Sol Dudak
            # 291   : Sag Dudak
            # 199   : Cene
            # Pozisyon tespitinde kullanilacak landmark moktalarinin edilmesi
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    
                    x = lm.x * img_w
                    y = lm.y * img_h

                    # Burun
                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)
                    
                    # Get the 2D coordinates
                    face_2d.append([(x),(y)])

                    # Get the 2D coordinates
                    face_3d.append([(x),(y),lm.z])

                    cv2.putText(image, str(idx), ( int(x), int(y) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                    cv2.circle(image, ( int(x), int(y) ), 5, (225, 0, 100), -1)
            
       
            #Convert it to np array
            face_2d = np.array(face_2d, dtype=np.float64)

            #Convert it to np array
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera internals

            size = image.shape

            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double")
            
            
            
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_coeffs)
            
            print(f"Rotation Vector:\n {0}".format(rotation_vector))
            print (f"Translation Vector:\n {0}".format(translation_vector))
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            nose_end_point2D, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            print("3d:")
            print(nose_end_point2D)

            for p in face_2d:
                cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            p1 = ( int(face_2d[0][0]), int(face_2d[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
 
            cv2.line(image, p1, p2, (255,0,0), 3)
    
    # Display image
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

while True:
    if cv2.waitKey(1) & 0xFF == ord('a') :
        cap.release()
        cv2.destroyAllWindows()