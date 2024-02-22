import cv2

cap = cv2.VideoCapture(-1)  # Open the first camera connected to the computer.

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter('captured_video/output.avi', fourcc, 20.0, (640, 480))

while cv2.waitKey(1) < 0:
    ret, frame = cap.read()  # Read an image from the frame.
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA) # Resize frame
    output.write(frame) # Write frame to output file

    cv2.imshow('frame', frame)

cap.release()
cap.release()
cv2.destroyAllWindows()
