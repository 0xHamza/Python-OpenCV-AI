
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')


import mss

while True:
    # Get the screenshot
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[1])

    # Resize the image
    img = cv2.resize(np.array(img), (640, 480))

    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Screenshot", np.array(img))

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window
cv2.destroyAllWindows()
'''
from PIL import ImageGrab
while True:
    # Get the screenshot
    img = ImageGrab.grab()

    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Screenshot", np.array(img))

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window
cv2.destroyAllWindows()
'''
