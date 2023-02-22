

import cv2


def zoom(image, scale=50):

    # get the webcam size
    height, width, channels = image.shape

    original = image.copy()

    cv2.putText(frame, "Width Height:"+str([width, height]), (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)
    
    # prepare the crop
    centerY, centerX = int(height/2), int(width/2)
    radiusY, radiusX = int(scale*height/100), int(scale*width/100)
    cv2.putText(frame, "Center:"+str([centerX, centerY]), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)

    minX, maxX = centerX-radiusX, centerX+radiusX
    minY, maxY = centerY-radiusY, centerY+radiusY
    cv2.putText(frame, "Radius:"+str([radiusX, radiusY]), (15,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)

    cv2.circle(frame,(centerX,centerY),3,(0,0,255),-1)
    cv2.rectangle(frame, (minX, minY), (maxX, maxY), (255, 0, 0), 2)
    cv2.putText(frame, "Scale: "+str(scale)+"x - "+str([minX,minY, maxX, maxY]), (15,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)
    cv2.imshow("Scaling"+str(scale)+"x", frame)

    cropped = original[minY:maxY , minX:maxX]
    resized_cropped = cv2.resize(cropped, (width, height))

    return resized_cropped


# kamera bağlantısını aç
cap = cv2.VideoCapture(0)

while True:
    # kameradan gray görüntü al
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("Video Original", frame)

    cv2.imshow("Video Zoom IN ", zoom(frame, 25))

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# kamera bağlantısını kapat
cap.release()
cv2.destroyAllWindows()