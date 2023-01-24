# This programs calculates the orientation of an object.
# The input is an image, and the output is an annotated image
# with the angle of otientation for each object (0 to 180 degrees)
 
import cv2 as cv,cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
 
# Load the image
image = cv.imread("opencv/data/hamza/profil.png")


def predict(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)
        
        # Ignore contours that are too small or too large
        if area < 3000 or 100000 < area:
            continue

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        if width < height:
            angle = 90 - angle
        else:
            angle = -angle

        label = "Rot. Angle:" + str(angle)
        
        cv.putText(img, label, box[0], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)


def videodanAlgila():
    # kamera bağlantısını aç
    cap = cv2.VideoCapture(0)

    while True:
        # kameradan gray görüntü al
        ret, frame = cap.read()
       
        predict(frame)

        # görüntüyü ekranda göster
        cv2.imshow("Video", frame)

        # 'q' tuşuna basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    # kamera bağlantısını kapat
    cap.release()
    cv2.destroyAllWindows()


videodanAlgila()