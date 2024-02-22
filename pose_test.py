from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

import cv2

img = cv2.imread("test.jpg")

body_parts = [
    'nose',
    'left eye (inner)',
    'left eye',
    'left eye (outer)',
    'right eye (inner)',
    'right eye',
    'right eye (outer)',
    'left ear',
    'right ear',
    'mouth (left)',
    'mouth (right)',
    'left shoulder',
    'right shoulder',
    'left elbow',
    'right elbow',
    'left wrist',
    'right wrist',
    'left pinky',
    'right pinky',
    'left index',
    'right index',
    'left thumb',
    'right thumb',
    'left hip',
    'right hip',
    'left knee',
    'right knee',
    'left ankle',
    'right ankle',
    'left heel',
    'right heel',
    'left foot index',
    'right foot index'
]




def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
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


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("test.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

pose_landmarks_list = detection_result.pose_landmarks

landmark_locations = {}

with open('body_part_locations.txt', 'w') as file:
  file.write("Locations (xyz) grouped by body part\n")
  for i in range(0,len(pose_landmarks_list[0])):
    current_body_part = body_parts[i]
    current_landmark = pose_landmarks_list[0][i]
    current_location = (current_landmark.x, current_landmark.y, current_landmark.z)
    file.write(f"{current_body_part}: {current_location}\n")

print(pose_landmarks_list[0][12])
print(pose_landmarks_list[0][11])

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
while cv2.waitKey(1) < 0:
  cv2.imshow('', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#cv2.waitKey(1)


