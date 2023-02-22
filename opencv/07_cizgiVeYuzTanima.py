import cv2
import numpy as np

# yüz tanıma algoritmasının eğitilmiş modeli
face_cascade = cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data\haarcascades\haarcascade_eye.xml')

def drawTick(img,x,y,w,h):
    cv2.line(img, (x, y+(h//2)), (x+(w//2),y+h), (255,255,0), 2)
    cv2.line(img, (x+(w//2),y+h), (x+w,y), (255,255,0), 2)

def drawCross(img,x,y,w,h):
    cv2.line(img, (x+w, y), (x,y+h), (0,0,0),2)
    cv2.line(img, (x, y), (x+w,y+h), (0,0,0),2)

def resimdenAlgila(path):
    # resmi oku
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # yüzleri algıla
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    adt = f"Adet: {len(faces)}"
    cv2.putText(image, adt, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # yüzleri kutu içine a2
    for (x, y, w, h) in faces:
        drawTick(image,x,y,w,h)
        
        xywh = f"x: {x} y: {y} w: {w} h: {h}"
        cv2.putText(image, xywh, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # resmi göster
    cv2.imshow("Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def videodanAlgila():
    # kamera bağlantısını aç
    cap = cv2.VideoCapture(0)

    while True:
        # kameradan gray görüntü al
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # yüzleri algıla
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        adt = f"Adet: {len(faces)}"
        cv2.putText(frame, adt, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)

        # yüzleri kutu içine al
        for (x, y, w, h) in faces:
            xywh = f"x: {x} y: {y} w: {w} h: {h}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, xywh, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)
            
            drawTick(frame,x,y,w,h)

        # görüntüyü ekranda göster
        cv2.imshow("Video", frame)

        # 'q' tuşuna basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    # kamera bağlantısını kapat
    cap.release()
    cv2.destroyAllWindows()

# resim dosyasının yolu
image_path = "data/hamza/profil.png"

# RESIMDEN ALGILAMA
resimdenAlgila(image_path)

# VIDEODAN ALGILAMA
videodanAlgila()