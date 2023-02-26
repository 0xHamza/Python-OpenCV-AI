# Hamza CELIK, Resim Islemleri
import cv2 as cv
import numpy as np 

# Read image
img = cv.imread('opencv/data/hamza/profil.png')

# Resim boyut
height, width , _c = img.shape
img = cv.resize(img,(int(width*0.85),int(height*0.85)))
print(f"Resim Boyut: {img.shape}")

# Convert the img to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply edge detection method on the image
edges = cv.Canny(gray, 50, 150)

# İkili görüntü oluşturma
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# Bulanıklaştırma
blur = cv.blur(img, (5,5))

# Ayna görüntüsü oluşturma
flip = cv.flip(img, 1)

cv.imshow('orjinal',img)
cv.imshow('gray',gray)
cv.imshow('edges',edges)
cv.imshow('Binary', binary)
cv.imshow('Blur', blur)
cv.imshow('Flip', flip)
cv.waitKey(0)
cv.destroyAllWindows()
