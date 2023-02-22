import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from ekFonksiyonlar import *


mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec_black = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,0))
drawing_spec_red = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(255,255,255))

face_mesh = mp_face_mesh.FaceMesh( 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


            # 473    : Sag Goz Mercegi Merkez
            # 474    : Sag Goz Mercek Sag Kose
            # 475    : Sag Goz Mercek Ust Kose
            # 476    : Sag Goz Mercek Sol Kose
            # 477    : Sag Goz Mercek Alt Kose

            # 468    : Sol Goz Mercegi Merkez
            # 469    : Sol Goz Mercek Sag Kose
            # 470    : Sol Goz Mercek Ust Kose
            # 471    : Sol Goz Mercek Sol Kose
            # 472    : Sol Goz Mercek Alt Kosez

            # 33    : Sol Goz Sol Kose
            # 133   : Sol Goz Sag Kose
            # 362   : Sag Goz Sol Kose
            # 263   : Sag Goz Sag Kose
            # 1     : Burun
            # 61    : Sol Dudak
            # 291   : Sag Dudak
            # 199   : Cene

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]


cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()

    start = time.time()
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB

    image = zoom(image, 25)

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

    goz_sol = [[0,0],[0,0],[0,0]]
    goz_sag = [[0,0],[0,0],[0,0]]
 
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Pozisyon tespitinde kullanilacak landmark moktalarinin edilmesi
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 470 or idx == 471 or idx == 468 or idx == 469 or idx == 472:

                    x = lm.x * img_w
                    y = lm.y * img_h

                    # Burun
                    if idx == 468:
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)
                    
                    # SOL GOZ
                    if idx == 471:
                        left_eye_2d = (x, y)

                    # SAG GOZ
                    if idx == 469:
                        right_eye_2d = (x, y)

                    # Get the 2D coordinates
                    face_2d.append([int(x),int(y)])

                    # Get the 2D coordinates
                    face_3d.append([int(x),int(y),lm.z])

                if idx == 33 or idx == 133 or idx == 263 or idx == 362 or idx == 473 or idx == 468 or idx == 34 or idx == 41:
                    x = lm.x * img_w
                    y = lm.y * img_h
                    cv2.putText(image, str(idx), ( int(x), int(y) ), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.putText(image, str(int(x))+","+str(int(y)), ( int(x), int(y)-15 ), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.circle(image, ( int(x), int(y) ), 1, (225, 0, 100), -1)

                    # SOL GOZ
                    if idx == 33:
                        goz_sol[0]=[int(x),int(y)]
                    elif idx == 133:
                        goz_sol[1]=[int(x),int(y)]
                    elif idx == 468:
                        goz_sol[2]=[int(x),int(y)]
                    # SAG GOZ
                    elif idx == 362:
                        goz_sag[1]=[int(x),int(y)]
                    elif idx == 263:
                        goz_sag[0]=[int(x),int(y)]
                    elif idx == 473:
                        goz_sag[2]=[int(x),int(y)]

                if idx < 500 and idx >490:                
                    x = lm.x * img_w
                    y = lm.y * img_h
                    cv2.putText(image, str(idx), ( int(x), int(y) ), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.circle(image, ( int(x), int(y) ), 1, (225, 0, 100), -1)

            cv2.line(image,goz_sol[0],goz_sol[1],(0,255,0),1)
            cv2.line(image,goz_sag[0],goz_sag[1],(0,255,0),1)

            dist_sol = distance(goz_sol[0],goz_sol[1]) # 2 Noktanin(x,y)(x2,y2) ÖKLİD Uzakligi
            dist_sag = distance(goz_sag[0],goz_sag[1]) # 2 Noktanin ÖKLİD Uzakligi

            x1, y1 = goz_sol[0]
            x2, y2 = goz_sol[1]
            '''m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            x_center = int((x1 + x2) / 2)
            y_center = int(m * x_center + b)    
            sol_merkez = (x_center, y_center)
            '''
            sol_merkez = (int((x1+x2)/2), int((y1+y2)/2))

            x1, y1 = goz_sag[0]
            x2, y2 = goz_sag[1]
            '''
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            x_center = int((x1 + x2) / 2)
            y_center = int(m * x_center + b)
            sag_merkez = (x_center, y_center)
            '''
            sag_merkez = (int((x1+x2)/2), int((y1+y2)/2))

            '''
            sol_width_center, sag_width_center= int(dist_sol/2), int(dist_sag/2)
            sol_merkez = (goz_sol[0][0]+sol_width_center,goz_sol[0][1])
            sag_merkez = (goz_sag[0][0]+sag_width_center,goz_sag[0][1])
            '''
            
            sol_pupilla = goz_sol[2]
            sag_pupilla = goz_sag[2]
            
            sol_pupilla_merkez_mesafe = distance(sol_merkez, sol_pupilla)
            sag_pupilla_merkez_mesafe = distance(sag_merkez, sag_pupilla)

            sol_yakinlik_genislik = sol_pupilla[0] - sol_merkez[0]
            sol_yakinlik_yukseklik = sol_pupilla[1] - sol_merkez[1]



            cv2.putText(image, str(int(dist_sol)), (sol_merkez[0]-5,sol_merkez[1]+25) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.putText(image, str(int(dist_sag)), (sag_merkez[0]-5,sag_merkez[1]+25) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            cv2.putText(image, str(sol_merkez), (sol_merkez[0]-15,sol_merkez[1]+45) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.putText(image, str([sol_yakinlik_genislik,sol_yakinlik_yukseklik]), (sol_merkez[0]-15,sol_merkez[1]+65) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.putText(image, str(sag_merkez), (sag_merkez[0]-15,sag_merkez[1]+45) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            
            cv2.line(image,(sol_merkez[0],sol_merkez[1]-5),(sol_merkez[0],sol_merkez[1]+5),(255,255,255),1)
            cv2.line(image,(sag_merkez[0],sag_merkez[1]-5),(sag_merkez[0],sag_merkez[1]+5),(255,255,255),1)

            if sol_yakinlik_genislik < 0 and sol_yakinlik_yukseklik < 0:
                msg = "Sol Ust"
            elif sol_yakinlik_genislik < 0 and sol_yakinlik_yukseklik > 0:
                msg = "Sol Alt"
            elif sol_yakinlik_genislik > 0 and sol_yakinlik_yukseklik > 0:
                msg = "Sag Alt"
            elif sol_yakinlik_genislik > 0 and sol_yakinlik_yukseklik < 0:
                msg = "Sag Ust"
            else:
                msg = "Duz"


            #Convert it to np array
            
            face_2d = np.array(face_2d, dtype=np.float64)

            #Convert it to np array
            face_3d = np.array(face_3d, dtype=np.float64)

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
            cv2.putText (image, msg, (20, 50), cv2. FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0,255),2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0,255),2)

        end = time.time()
        totalTime = end-start

        fps = 1 / totalTime
        #print("FPS: ",fps)

        cv2.putText(image, f"FPS: {int (fps)}", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
     
        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec_red)
        
    else:
        cv2.putText (image, "YUZ BULUNAMADI!", (20, 50), cv2. FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    print("-"*25)
    print("FACE 2D:\n",face_2d)
    print("FACE 3D:\n",face_3d)
    
    cv2.imshow("Head Pose Estimation", image)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        plt.subplot(),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Input')
        plt.show()
        break


while True:
    if cv2.waitKey(1) & 0xFF == ord('a') :
        cap.release()
        cv2.destroyAllWindows()
        break