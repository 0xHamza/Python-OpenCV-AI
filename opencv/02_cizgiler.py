# importing cv2 
import cv2 
   
# path 
path = 'data\images\geeks14.png'
   
# Reading an image in grayscale mode
image = cv2.imread(path, 0)
   
# Window name in which image is displayed
window_name = 'Image'
  

# çizgi çizmek için başlangıç ve bitiş koordinatları

width = image.shape[1] 
height = image.shape[0]

start_point = (width // 2, 0)
end_point = (width // 2, height)

# Black color in BGR
color = (0, 0, 0)
  
# Line thickness of 5 px
thickness = 5
  
# Using cv2.line() method
# Draw a diagonal black line with thickness of 5 px
image = cv2.line(image, start_point, end_point, color, thickness)
image = cv2.line(image, (20,20), (width-20,height-20), color, thickness)
image = cv2.line(image, (width-20,20), (20,height-20), color, thickness)

# çizilen çizgili görüntüyü ekranda göster
cv2.imshow("Image with Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()