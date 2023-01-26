
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('opencv\data\haarcascades\haarcascade_frontalface_default.xml')
import mss
import pyautogui

while True:
    # Get the mouse position
    x, y = pyautogui.position()

    # Get the region of the screen around the mouse
    with mss.mss() as sct:
        img = sct.grab({"left": x-150, "top": y-150, "width": 300, "height": 300})

    image = np.array(img).copy()
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    adt = f"Adet: {len(faces)}"
    cv2.putText(image, adt, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #center face
        center = (x + int(w/2), y + int(h/2))
        cv2.circle(image, center, 10, (0, 0, 255), -1)

    # Display the output
    cv2.imshow("Screenshot", np.array(image))

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("son_goruntu.jpg", image) # frame, son görüntü olarak kaydedilir.
        break

# Close the window
cv2.destroyAllWindows()