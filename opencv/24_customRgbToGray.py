import cv2

# kamera bağlantısını aç
cap = cv2.VideoCapture(0)

# kameradan gray görüntü al
ret, image = cap.read()
orjinal = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

width, height, _ = image.shape

for i in range(width):
  for j in range(height):
    b = image[i, j][0]
    g = image[i, j][1]
    r = image[i, j][2]

    # NEGATIF ALMA
    _b = 255 - b
    _g = 255 - g
    _r = 255 - r

    # GRAY
    gr = int((r+g+b)/3)
    gr =  0.299 * r + 0.587 * g + 0.114 *b

    image[i, j][0] = gr
    image[i, j][1] = gr
    image[i, j][2] = gr

cv2.imshow('GRAY CUSTOM', cv2.flip(image, 1))
cv2.imshow('GRAY OPENCV', cv2.flip(gray, 1))
cv2.imshow('ORJINAL', cv2.flip(orjinal, 1))

while True:
  if cv2.waitKey(5) & 0xFF == 27:
    cap.release()

