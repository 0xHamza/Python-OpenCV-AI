# Hamza CELIK

import cv2
import numpy as np

# 300x300 boyutunda beyaz bir resim oluştur
img = np.full((300, 300, 3), 255, dtype=np.uint8)

# Kare (resim, baslangic_X_Y, bitis_X_Y, renk, kalinlik)
cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 0), 2)

# Cember (resim, merkezXY, yariCap, renk, dolu-bos)
cv2.circle(img, (150, 150), 75, (0, 255, 0), -1)
cv2.circle(img, (150, 150), 100, (0, 255, 0), 1)

# ELips (resim, merkezXY, genislik, yukseklik, donmeAcisi, basDerece, sonDerece, renk, kalinlik)
cv2.ellipse(img, (150, 275), (50, 25), 0, 0, 360, (255, 255, 0), 2)
cv2.ellipse(img, (150, 275), (35, 15), 0, 0, 360, (255, 255, 0), -1)

# Cizgi (resim, baslangicXY, bitisXY, renk, kalinlik)
cv2.line(img, (0, 0), (50, 50), (0, 0, 255), 2)
cv2.arrowedLine(img, (300, 0), (250, 50), (0, 0, 255), 2) # Cizgi Ok

# Üçgen için nokta koordinatlarını belirle
pts = np.array([ [125, 50], [175, 50],[150, 0]])
pts2 = np.array([ [135, 40], [165, 40],[150, 15]])
cv2.polylines(img, [pts], True, (0, 0, 255), 2) # Noktalari birlesitir
cv2.fillPoly(img, [pts2], (0, 0, 255)) # Noktalarin icini doldur

# Yazi (resim, 'metin', konum_XY, font, buyukluk, renk, kalinlik, stil)
text = "Hamza"
font1 = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
stil = cv2.LINE_AA
cv2.putText(img, text, (100,125), font1, 1 , (0, 0, 0), 1)
cv2.putText(img, 'CELIK', (100,160), font1, 1 , (0, 0, 0), 2,stil)
cv2.putText(img, text, (100,195), font2, 1 , (0, 0, 0), 1,stil)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()