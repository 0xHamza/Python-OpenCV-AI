import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('opencv\data\haarcascades\haarcascade_frontalface_default.xml')

import mss
import pyautogui

frame_resolution = 300
screen_width, screen_height = pyautogui.size()

while True:
    # Get the mouse position
    x, y = pyautogui.position()

    # Get the region of the screen around the mouse
    with mss.mss() as sct:
        img_pos_x = x-150
        img_pos_y = y-150
        img = sct.grab({"left": img_pos_x, "top": img_pos_y, "width": 300, "height": 300})

    image = np.array(img).copy()
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    adt = f"Adet: {len(faces)}"
    pos_mouse = f"XY: ({x},{y})"

    cv2.putText(image, adt, (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(image, pos_mouse, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        
        #center face
        center_x = x + int(w/2)
        center_y = y + int(h/2)
        center_face = (center_x, center_y)
        pos_face = f"F. XY: ({center_x},{center_y})"
        cv2.putText(image, pos_face, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(image, center_face, 10, (0, 0, 255), -1)

        pyautogui.moveTo(img_pos_x+center_x,img_pos_y+center_y)

    # Display the output
    
    cv2.imshow("Screenshot", np.array(image))
    

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("son_goruntu.jpg", image) # frame, son görüntü olarak kaydedilir.
        break

# Close the window
cv2.destroyAllWindows()