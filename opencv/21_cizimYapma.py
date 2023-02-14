import cv2
import numpy as np

drawSS = False
# Mouse kontrol fonkisyonu(mouse callback function)
def draw_circle(event,x,y,flags,param):
    global drawSS
    if event == cv2.EVENT_MOUSEMOVE and drawSS == True:
        cv2.circle(img,(x,y),5,(255,0,0), 1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(img,(x,y),(x+25,y+25),(255,0,255),-1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        drawSS = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawSS = False

# Siyah bir panel oluştur ve fonksiyonu birleştir
img = np.zeros((512,512,3), dtype=np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image' , img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()