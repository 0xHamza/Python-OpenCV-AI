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
            
            # Pozisyon tespitinde kullanilacak landmark moktalarinin edilmesi
            # 33    : Sol Goz
            # 263   : Sag Goz 
            # 1     : Burun
            # 61    : Sol Dudak
            # 291   : Sag Dudak
            # 199   : Cene

            LM_BURUN = 1
            LM_GOZ_SOL = 33
            LM_GOZ_SAG = 263
            LM_DUDAK_SOL = 61
            LM_DUDAK_SAG = 291
            LM_CENE = 199

            POINTS = [LM_BURUN,LM_GOZ_SOL, LM_DUDAK_SOL,LM_CENE,LM_GOZ_SAG,LM_DUDAK_SAG]

            #all_landmarks =  [lm for lm in face_landmarks.landmark]
            landmarks_points = np.array([ np.multiply([lm.x,lm.y,lm.z],[img_w,img_h,1]) for lm in face_landmarks.landmark ])


            nose_2d = landmarks_points[LM_BURUN][:2]
            nose_3d = np.hstack((landmarks_points[LM_BURUN][:2], [landmarks_points[LM_BURUN][2]*3000]))
            nose_3d = (landmarks_points[LM_BURUN][0], landmarks_points[LM_BURUN][1],landmarks_points[LM_BURUN][2]*3000)

            left_eye_2d = landmarks_points[LM_GOZ_SOL][:2]

            right_eye_2d = landmarks_points[LM_GOZ_SAG][:2]
            
            # Get the 2D coordinates
            _3d = [[int(landmarks_points[point][0]),int(landmarks_points[point][1]),landmarks_points[point][2]] for point in POINTS]
            face_3d = np.array(_3d, np.float64)
            _2d = face_3d[:,:2]
            face_2d = np.array(_2d, dtype=np.float64)

            for p in POINTS:
                x,y = landmarks_points[p][0],  landmarks_points[p][1]
                cv2.putText(image, str(p), ( int(x), int(y) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                cv2.circle(image, ( int(x), int(y) ), 5, (225, 0, 100), -1)
    

            #The cameta matrix
            focal_length = 1*img_w

            cam_matrix= np.array([  [focal_length,  0,              img_h/2],
                                    [0,             focal_length,   img_w/2],
                                    [0,             0,              1]])


            #The distortion parameters
            dist_matrix = np.zeros( (4,1), dtype=np.float64)

            # Solve PnP, POZISYON TAHMIN METODU
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
                if x < -10:
                    text += " Down"
                elif x > 10:
                    text += " Up"
            elif y > 10:
                text = "Looking Right"
                if x < -10:
                    text += " Down"
                elif x > 10:
                    text += " Up"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 255, 0), 3)

            p1 = (int(left_eye_2d[0]), int(left_eye_2d[1]))
            p2 = (int(left_eye_2d[0] + y * 10), int(left_eye_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 255, 0), 3)

            p1 = (int(right_eye_2d[0]), int(right_eye_2d[1]))
            p2 = (int(right_eye_2d[0] + y * 10), int(right_eye_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 255, 0), 3)

            # Add the text on the image
            cv2.putText (image, text, (20, 50), cv2. FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0,255),2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0,255),2)

        end = time.time()
        totalTime = end-start

        fps = 1 / totalTime
        #print("FPS: ",fps)

        cv2.putText(image, f"FPS: {int (fps)}", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
        mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS
                    ,landmark_drawing_spec = drawing_spec
                    ,connection_drawing_spec = drawing_spec
                    )

    #print(face_2d)
    #print(face_3d)

    cv2.imshow("Head Pose Estimation", image)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break


while True:
    if cv2.waitKey(1) & 0xFF == ord('a') :
        cap.release()
        cv2.destroyAllWindows()