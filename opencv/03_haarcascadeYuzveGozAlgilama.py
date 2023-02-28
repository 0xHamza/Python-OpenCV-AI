import cv2

# Haar Cascade sınıflandırıcısını yükleme
face_cascade = cv2.CascadeClassifier('opencv/data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv/data\haarcascades\haarcascade_eye.xml')

# Resmi yükleme
img = cv2.imread('opencv/data/hamza/profil.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Yüz tespiti yapma
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
# Goz tespiti yapma
eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

# Yüzleri çizme
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Gozleri çizme
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

adtYuz = f"Yuz Sayisi: {len(faces)}"
adtGoz = f"Goz Sayisi: {len(eyes)}"
fnt = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, adtYuz, (15, 15), fnt, 0.5, (0,0,0), 1)
cv2.putText(img, adtGoz, (15, 35), fnt, 0.5, (0,0,0), 1)
cv2.imshow('Haar Cascade', img)
cv2.waitKey()



