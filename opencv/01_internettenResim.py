# Hamza CELIK, Resim Okuma Internetten
import numpy as np
import cv2
import urllib.request

resim = 'https://www.dpu.edu.tr/app/views/panel/ckfinder/userfiles/1/images/logolar/dpu-logo1.png'

def internetten(url):
	resp = urllib.request.urlopen(url)                          # Resmi indirme
	image = np.asarray(bytearray(resp.read()), dtype="uint8")   # Resmi okuyup np turune donusturme
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)               # Resmi opencv ile okuma
	return image

# Internette oku ve yendşen boyutlandir
img = cv2.resize(internetten(resim), (200,200)) 

# Pencereyi göster
cv2.imshow('Resim Internetten', img)

# Kullanıcının tuşa basmasını bekler
cv2.waitKey(0)

# Pencereyi kapatır
cv2.destroyAllWindows()

