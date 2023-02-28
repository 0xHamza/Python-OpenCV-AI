import cv2
import numpy as np

# Egitim dosyalarının bulunduğu dizin
model = 'opencv/data/train_dataset/trained_model_lbph.xml'
names_saved_path = 'opencv/data/train_dataset/trained_model_lbph_names.txt'

# Kayitlarin etiketlerin isimlerini txt dosyadan array olarak okuma
names = np.genfromtxt(names_saved_path, dtype='str')

# Yuz Tanima Modeli yükle
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model)

# Yuz algilama modeli
face_cascade = cv2.CascadeClassifier('opencv/data\haarcascades\haarcascade_frontalface_default.xml')

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