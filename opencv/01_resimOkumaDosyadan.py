# Hamza CELIK, Resim Okuma Dosyadan

# Ekle kutupahneyi, yoksa: 'pip install opencv'
import cv2

# Resmi yükle
img = cv2.imread('opencv/data/hamza/profil.png')

# Pencereyi göster
cv2.imshow('Resim Dosyadan', img)

# Kullanıcının tuşa basmasını bekler
cv2.waitKey(0)

# Pencereyi kapatır
cv2.destroyAllWindows()