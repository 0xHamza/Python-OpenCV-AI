import cv2 as cv,cv2
import numpy as np 

# Read image
img = cv2.imread('opencv/data/hamza/profil.png')
  
# Convert the img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Apply edge detection method on the image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

shp = img.shape
print(f"Resim Boyut: {shp}")

cv.imshow('img',img)
cv.imshow('gray',gray)
cv.imshow('edges',edges)
cv.waitKey(0)
cv.destroyAllWindows()
