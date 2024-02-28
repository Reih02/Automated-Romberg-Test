import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from filterpy.kalman import KalmanFilter
import math


video_path = 'captured_video/romberg_positive.mp4'
vid = cv2.VideoCapture(video_path)

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


def draw_landmarks_on_image(rgb_image, detection_result, computed_cog, unbalanced, right_foot, left_foot, smoothed_pos_right):
  #print(f"COMPUTED COG IS : {computed_cog}")
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  #print(pose_landmarks_list[0])
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
  # draw COG
  height, width, _ = annotated_image.shape
  x_pixel_location = int(computed_cog[0] * width)
  y_pixel_location = int(computed_cog[1] * height)
  cv2.circle(annotated_image, (x_pixel_location, y_pixel_location), 10, (255, 0, 0), 2)

  # draw lines between foot and COG
  right_foot_x, right_foot_y = int(right_foot[0] * width), int(right_foot[1] * height)
  left_foot_x, left_foot_y = int(left_foot[0] * width), int(left_foot[1] * height)
  cv2.line(annotated_image, (right_foot_x, right_foot_y), (x_pixel_location, y_pixel_location), (120, 0, 120), 2)
  cv2.line(annotated_image, (left_foot_x, left_foot_y), (x_pixel_location, y_pixel_location), (120, 0, 120), 2)

  smoothed_right_x, smoothed_right_y = int(smoothed_pos_right[0] * width), int(smoothed_pos_right[1] * height)
  cv2.circle(annotated_image, (smoothed_right_x, smoothed_right_y), 3, (0, 255, 0), 2)

  # draw smoothed COG
  #smooth_x_pixel_location = int(smoothed_cog[0] * width)
  #smooth_y_pixel_location = int(smoothed_cog[1] * height)
  #cv2.circle(annotated_image, (smooth_x_pixel_location, smooth_y_pixel_location), 15, (0, 0, 255), 2)

  if unbalanced:
    cv2.circle(annotated_image, (50, 50), 5, (0, 0, 255), 5)

  

  return annotated_image

# computes the weighted centre of gravity of subject + plots this on the image
def compute_cog():
  
  x_centre = 0
  y_centre = 0  

  # hard-coded anatomically accurate joint weights w/ normalisation for use in later calculation
  weights = {'head': 0.0681/0.5336, 'UPT': 0.1571/0.5336, 'upper_arm': 0.0263/0.5336,
                     'hand': 0.0059/0.5336, 'LPT': 0.1182/0.5336, 'thigh': 0.1447/0.5336, 'foot': 0.0133/0.5336}
  bodyparts_classifications = {'nose': 'head', 'left shoulder': 'UPT', 'right shoulder': 'UPT',
                               'left elbow': 'upper_arm', 'right elbow': 'upper_arm',
                               'left wrist': 'hand', 'right wrist': 'hand', 'left hip': 'LPT',
                               'right hip': 'LPT', 'left knee': 'thigh', 'right knee': 'thigh',
                               'left ankle': 'foot', 'right ankle': 'foot'}
  
  with open('body_part_locations.txt', 'r') as file:
    lines = file.readlines()[1:]
    joint_locations = []
    for line in lines:
        parts = line.split(':')
        body_part = parts[0].strip()
        location_str = parts[1].strip()
        coordinates = location_str[1:-1].split(',')
        coordinates = tuple(float(coord.strip()) for coord in coordinates)
        joint_locations.append((body_part, coordinates))
        
    # iterate through the relevant joints and move the calculated centre of mass 
    # according to the joint location and the appropriate anatomical weight 
    used_joints = 0
    for i in range(0, len(joint_locations)):
       if joint_locations[i][0] in bodyparts_classifications:
         current_bodypart = bodyparts_classifications[joint_locations[i][0]]
         x_centre += (weights[current_bodypart] * joint_locations[i][1][0])
         y_centre += (weights[current_bodypart] * joint_locations[i][1][1])
         used_joints += weights[current_bodypart]
    x_centre /= used_joints
    y_centre /= used_joints
    return (x_centre, y_centre)
  
class SmoothCOG:
  def __init__(self, alpha):
    self.alpha = alpha
    self.smoothed_x = 0
    self.smoothed_y = 0
    self.used = False
  
  def smooth_data(self, current_cog):
    if not self.used:
      self.used = True
      self.smoothed_x = current_cog[0]
      self.smoothed_y = current_cog[1]
    else:
      self.smoothed_x = self.alpha * current_cog[0] + (1 - self.alpha) * self.smoothed_x
      self.smoothed_y = self.alpha * current_cog[1] + (1 - self.alpha) * self.smoothed_y

    return (self.smoothed_x, self.smoothed_y)
  
# Calculates how much relative weight is stored on each foot based on pose geometry and
# calculated centre of mass
def calculate_weight_distribution(cog):
  with open('body_part_locations.txt', 'r') as file:
    lines = file.readlines()[-2:]
    joint_locations = []
    for line in lines:
        parts = line.split(':')
        body_part = parts[0].strip()
        location_str = parts[1].strip()
        coordinates = location_str[1:-1].split(',')
        coordinates = tuple(float(coord.strip()) for coord in coordinates)
        joint_locations.append(coordinates[0:2])

    right_foot, left_foot = joint_locations[0], joint_locations[1]
    distance_right_foot = math.sqrt((right_foot[0] - cog[0])**2 + (right_foot[1] - cog[1])**2)
    distance_left_foot = math.sqrt((left_foot[0] - cog[0])**2 + (left_foot[1] - cog[1])**2)

    weight_distro_right = (1 / distance_right_foot) * (1 / (distance_right_foot + distance_left_foot)) * 100
    weight_distro_left = (1 / distance_left_foot) * (1 / (distance_right_foot + distance_left_foot)) * 100

    # normalise vals
    #right_ratio = distance_right_foot / (distance_left_foot + distance_right_foot)
    #left_ratio = distance_left_foot / (distance_left_foot + distance_right_foot)

    # print(f"WEIGHT DISTRO RIGHT: {right_ratio}")
    # print(f"WEIGHT DISTRO LEFT: {left_ratio}")

    return (weight_distro_right, weight_distro_left, right_foot, left_foot)
  
def setup_kalman_r():
  # Define Kalman Filter
  kf = KalmanFilter(dim_x=4, dim_z=2)

  # Initialize state transition matrix A
  kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

  # Initialize measurement function H
  kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

  # Initialize measurement noise covariance matrix
  kf.R *= 100

  # Initialize process noise covariance matrix
  kf.Q = np.array([[0.1, 0,    0,    0],
                 [0,    0.1, 0,    0],
                 [0,    0,    0.3, 0],
                 [0,    0,    0,    0.3]])
  
  kf.x = np.array([0., 0., 0., 0.])  # initial state (x, y, vx, vy)
  kf.P = np.eye(4)                   # initial uncertainty

  return kf


  

# Create a PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
  
# Init COG smoother class with defined alpha val
alpha = 0.5
data_smoother = SmoothCOG(alpha)

patient_was_unbalanced = False

max_ratio_difference = 0

kf_r = setup_kalman_r()

# Load the input frames from the video.
while cv2.waitKey(1) < 0:
  ret, frame = vid.read()
  if not ret:
    break
  
  rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # convert video frame to mp.Image type 
                                                                     # for processing in pose detector
  
  detection_result = detector.detect(rgb_frame)

  pose_landmarks_list = detection_result.pose_landmarks

  if len(pose_landmarks_list) > 0:
    with open('body_part_locations.txt', 'w') as file:
      file.write("Locations (xyz) grouped by body part\n")
      for i in range(0,len(pose_landmarks_list[0])):
          current_body_part = body_parts[i]
          current_landmark = pose_landmarks_list[0][i]
          current_location = (current_landmark.x, current_landmark.y, current_landmark.z)
          file.write(f"{current_body_part}: {current_location}\n")

    computed_cog = compute_cog()

    #smoothed_cog = data_smoother.smooth_data(computed_cog)
    #print(f"RAW COG: {computed_cog[0], computed_cog[1]}")
    #print(f"SMOOTH COG: {smoothed_cog[0], smoothed_cog[1]}")

    right_ratio, left_ratio, right_foot, left_foot = calculate_weight_distribution(computed_cog)

    kf_r.predict()
    kf_r.update(right_foot)
    smoothed_pos_right = (kf_r.x[0], kf_r.x[1])

    print(f"Weight distribution difference is: {abs(right_ratio - left_ratio)}%")

    if abs(right_ratio - left_ratio) > max_ratio_difference:
      max_ratio_difference = abs(right_ratio - left_ratio)

    # check if subject is "unbalanced"
    if abs(right_ratio - left_ratio) > 6:
      print("###UNBALANCED###")
      patient_was_unbalanced = True
      unbalanced = True
    else:
      unbalanced = False

    #annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result, computed_cog, smoothed_cog, unbalanced, right_foot, left_foot)
    annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result, computed_cog, unbalanced, right_foot, left_foot, smoothed_pos_right)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", annotated_image)

vid.release()
cv2.destroyAllWindows()

print(f"Max ratio difference: {max_ratio_difference}")

print(f"Diagnosis: {patient_was_unbalanced}")

