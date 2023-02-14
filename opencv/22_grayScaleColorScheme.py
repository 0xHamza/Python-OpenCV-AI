import cv2 
import numpy as np

img = np.zeros((512,512,1), dtype=np.uint8)+255




size = 32

x,y = 0,0
n = 0

for i in range(0,16):
    for j in range(0,16):
        x = j * size
        y = i * size
        cv2.rectangle(img, (x,y),(x+size,y+size),(n),-1)
        cv2.putText(img, str(n),(x+8,y+20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255-n,255-n,255-n),1)
        n +=1

    

while(1):
    cv2.imshow('image' , img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()