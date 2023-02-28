import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('opencv/data\haarcascades\haarcascade_frontalface_default.xml')

# Resim dosyalarının bulunduğu dizin
dataset_path = 'opencv/data/train_dataset/'

# LBPH modelinin eğitildiği veri seti
names, faces, labels, label = [],[],[],0
for root, dirs, files in os.walk(dataset_path):
    name = os.path.basename(root)
    label +=1
    for file in files:
        img_path = os.path.join(root, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces_detector = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5);
        
        if (len(faces_detector) == 0): #YUZ ALGILANAMAZSA EGITME
            break

        (x, y, w, h) = faces_detector[0]
        detected = img[y:y+w, x:x+h]
        cv2.imshow("a",detected)
        faces.append(detected)
        labels.append(label)
        names.append(name)

# LBPH modelini oluşturma VE eGİTME
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# Kamera bağlantısını başlat
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(test_img, scaleFactor=1.1, minNeighbors=5)

    # Yüz tanıma
    for (x, y, w, h) in faces:
        roi = test_img[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{names[label-1]}, {confidence:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Sonuçları göster
    cv2.imshow('Frame', frame)
        
    # ESC tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) == 27:
        break
        
# Kamerayı serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()