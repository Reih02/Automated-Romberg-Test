import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


video_path = 'captured_video/sample_vid.mp4'

vid = cv2.VideoCapture(video_path)


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  print(pose_landmarks_list[0])
  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


# Create a PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Load the input frames from the video.
while cv2.waitKey(1) < 0:
  ret, frame = vid.read()
  if not ret:
    break
  
  rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # convert video frame to mp.Image type 
                                                                     # for processing in pose detector
  
  detection_result = detector.detect(rgb_frame)

  annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
  cv2.imshow('Frame', annotated_image)

vid.release()
cv2.destroyAllWindows()

