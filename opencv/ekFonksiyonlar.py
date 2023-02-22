import math
import cv2

def distance(A,B):
    # İki nokta arasındaki uzaklığı hesaplayın
    return math.sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)

def irisPositionState(oran):
    pos = ""
    if oran <=0.45:
        pos = "RIGHT"
    elif oran >0.45 and oran<=0.56:
        pos = "CENTER"
    else:
        pos = "LEFT"
    return pos

def irisPositionCalculate(center,right,left):
    total = distance(left,right)
    merkezden_saga = distance(center,right)
    merkezden_sola = distance(center,left)
    oran_sol = merkezden_sola/total
    oran_sag = merkezden_saga/total
    pos = irisPositionState(oran_sag)
    
    return pos,oran_sag,oran_sol
    
def zoom(image, scale=50):

    # get the webcam size
    height, width, channels = image.shape

    original = image.copy()

    cv2.putText(image, "Width Height:"+str([width, height]), (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)
    
    # prepare the crop
    centerY, centerX = int(height/2), int(width/2)
    radiusY, radiusX = int(scale*height/100), int(scale*width/100)
    cv2.putText(image, "Center:"+str([centerX, centerY]), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)

    minX, maxX = centerX-radiusX, centerX+radiusX
    minY, maxY = centerY-radiusY, centerY+radiusY
    cv2.putText(image, "Radius:"+str([radiusX, radiusY]), (15,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)

    cv2.circle(image,(centerX,centerY),3,(0,0,255),-1)
    cv2.rectangle(image, (minX, minY), (maxX, maxY), (255, 0, 0), 2)
    cv2.putText(image, "Scale: "+str(scale)+"x - "+str([minX,minY, maxX, maxY]), (15,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)
    cv2.imshow("Scaling"+str(scale)+"x", image)

    cropped = original[minY:maxY , minX:maxX]
    resized_cropped = cv2.resize(cropped, (width, height))

    return resized_cropped