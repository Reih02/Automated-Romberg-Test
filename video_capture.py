import cv2

cap = cv2.VideoCapture(-1)  # Open the first camera connected to the computer.

while cv2.waitKey(1) < 0:
    ret, frame = cap.read()  # Read an image from the frame.
    cv2.imshow('frame', frame)  # Show the image on the display.

# Release the camera device and close the GUI.
cap.release()
cv2.destroyAllWindows()
