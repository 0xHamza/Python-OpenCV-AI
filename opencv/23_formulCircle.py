import cv2 
import numpy as np
import math

img = np.zeros((512,512,1), dtype=np.uint8)




size = 32

#cember merkez
center_x,center_y = 256, 256
r = 256

for i in range(0,512):
    for j in range(0,512):

        if math.sqrt( pow((i-center_x),2) + pow((j-center_y),2)) < r:
            cv2.circle(img, (i,j),1,(255,255,255),1)
      

    

while(1):
    cv2.imshow('image' , img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()