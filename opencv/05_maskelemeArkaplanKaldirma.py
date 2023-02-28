import cv2

# MOG2 arka plan çıkarım algoritmasını oluştur
fgbg_MOG2 = cv2.createBackgroundSubtractorMOG2()

# KNN arka plan çıkarım algoritmasını oluştur
fgbg_KNN = cv2.createBackgroundSubtractorKNN()

fbgb = fgbg_MOG2

# video dosyasını oku
cap = cv2.VideoCapture(0)

while True:
    # her bir kare için
    ret, frame = cap.read()

    # arka planı çıkar
    fgmask = fbgb.apply(frame)

    # arka planı kaldırılmış kareyi ekranda göster
    cv2.imshow("Foreground", fgmask)

    # herhangi bir tuşa basılana kadar bekle
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()