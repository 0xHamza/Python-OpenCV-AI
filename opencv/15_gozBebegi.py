import cv2
import mediapipe as mp
import numpy as np
from ekFonksiyonlar import *

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe yüz ağı modeli yükleyin
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Kamerayı açın
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir çerçeve alın
    ret, frame = cap.read()

    height, width = frame.shape[:2]

    # Çerçeveden yüz landmark'larını belirleyin
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            #print(type(face_landmarks.landmark),face_landmarks.landmark,len(face_landmarks.landmark))

            all_landmarks =  [lm for lm in face_landmarks.landmark]
            landmarks_points = np.array([ np.multiply([lm.x,lm.y],[width,height]).astype(int) for lm in face_landmarks.landmark ])

            print(landmarks_points[1])

            # Sol göz pupil merkezini hesaplayın
            left_eye_pupil = landmarks_points[468]
            left_eye_sag_kose = landmarks_points[133]
            left_eye_sol_kose = landmarks_points[33]
            cv2.putText(frame, "SOL", (left_eye_pupil[0]-10,left_eye_pupil[1]-25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

            # Sağ göz pupil merkezini hesaplayın
            right_eye_pupil = landmarks_points[473] #(int(all_landmarks[473].x * frame.shape[1]), int(all_landmarks[473].y * frame.shape[0]))
            right_eye_sag_kose = landmarks_points[263]
            right_eye_sol_kose = landmarks_points[362]
            cv2.putText(frame, "SAG", (right_eye_pupil[0]-10,right_eye_pupil[1]-25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

            # Sol göz pupil merkezini göstermek için daire çizin
            cv2.circle(frame, left_eye_pupil, 2, (0, 0, 255), -1)
            cv2.circle(frame, left_eye_sag_kose, 2, (255, 0, 255), -1)
            cv2.circle(frame, left_eye_sol_kose, 2, (255, 255, 255), -1)

            # Sağ göz pupil merkezini göstermek için daire çizin
            cv2.circle(frame, right_eye_pupil, 2, (0, 0, 255), -1)
            cv2.circle(frame, right_eye_sag_kose, 2, (255, 0, 255), -1)
            cv2.circle(frame, right_eye_sol_kose, 2, (255, 255, 255), -1)
            
            l_pos, l_oranSag, l_oranSol = irisPositionCalculate(left_eye_pupil,left_eye_sag_kose,left_eye_sol_kose)
            r_pos, r_oranSag, r_oranSol = irisPositionCalculate(right_eye_pupil,right_eye_sag_kose,right_eye_sol_kose)

            oran = (l_oranSag+r_oranSag)/2
            ortak_pos = irisPositionState(oran)

            cv2.putText(frame, l_pos, (left_eye_pupil[0]-20,left_eye_pupil[1]+25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.putText(frame, r_pos, (right_eye_pupil[0]-20,right_eye_pupil[1]+25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.putText(frame, ortak_pos, (landmarks_points[1][0]-40,landmarks_points[1][1]+25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.putText(frame, f"Sol Goz: {l_pos} Oran:{l_oranSag:.2f}, {l_oranSol:.2f}", (25,25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
            cv2.putText(frame, f"Sag Goz: {r_pos} Oran:{r_oranSag:.2f}, {r_oranSol:.2f}", (25,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
            cv2.putText(frame, f"Ortak: {ortak_pos} Oran:{oran:.2f}", (25,75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

            if ortak_pos == "LEFT":
                cv2.putText(frame, "<", (int(width/2)-125,int(height/2)+30), cv2.FONT_HERSHEY_DUPLEX, 10, (0, 255, 0), 2)
            elif ortak_pos == "RIGHT":
                cv2.putText(frame, ">", (int(width/2)-125,int(height/2)+30), cv2.FONT_HERSHEY_DUPLEX, 10, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "-", (int(width/2)-125,int(height/2)+30), cv2.FONT_HERSHEY_DUPLEX, 10, (0, 255, 0), 2)

    # Kameradan gelen görüntüyü ekranda gösterin
    cv2.imshow('frame', frame)

    # Çıkış yapmak için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakın ve tüm pencereleri kapatın
cap.release()
cv2.destroyAllWindows()