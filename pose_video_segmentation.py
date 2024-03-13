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
import tensorflow as tf

video_path = 'captured_video/my_vid.MOV'
#other_vid = 'captured_video/weight.MOV'
vid = cv2.VideoCapture(video_path)
#vid2 = cv2.VideoCapture(other_vid)

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


def draw_landmarks_on_image(rgb_image, detection_result, computed_cog, unbalanced, right_foot, left_foot, smoothed_pos_right, smoothed_pos_left):
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
  # draw COG
  height, width, _ = annotated_image.shape
  x_pixel_location = int(computed_cog[0] * width)
  y_pixel_location = int(computed_cog[1] * height)
  cv2.circle(annotated_image, (x_pixel_location, y_pixel_location), 10, (255, 0, 0), 2)

  smoothed_right_x, smoothed_right_y = int(smoothed_pos_right[0] * width), int(smoothed_pos_right[1] * height)
  smoothed_left_x, smoothed_left_y = int(smoothed_pos_left[0] * width), int(smoothed_pos_left[1] * height)
  cv2.circle(annotated_image, (smoothed_right_x, smoothed_right_y), 3, (0, 255, 0), 2)
  cv2.circle(annotated_image, (smoothed_left_x, smoothed_left_y), 3, (0, 255, 0), 2)

  # draw lines between smoothed feed joint locations and COG
  cv2.line(annotated_image, (smoothed_right_x, smoothed_right_y), (x_pixel_location, y_pixel_location), (0, 0, 255), 2)
  cv2.line(annotated_image, (smoothed_left_x, smoothed_left_y), (x_pixel_location, y_pixel_location), (0, 0, 255), 2)

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
# https://physics.stackexchange.com/questions/805853/is-it-possible-to-calculate-the-weight-distribution-on-each-foot-from-the-centre/805861#805861
def calculate_weight_distribution(rgb_image, cog, smoothed_pos_right, smoothed_pos_left):
    image = np.copy(rgb_image)
    _, width, _ = image.shape

    x_right = smoothed_pos_right[0]
    x_left = smoothed_pos_left[0]

    cog_x = cog[0]

    x_right = int(x_right * width)
    x_left = int(x_left * width)
    cog_x = int(cog_x * width)
    
    
      
    x_1 = math.sqrt((x_right - cog_x) ** 2)
    x_2 = math.sqrt((x_left - cog_x) ** 2)

    if cog_x > x_left:
      x_2 *= -1
    if cog_x < x_right:
      x_1 *= -1
      
    N_1 = ((MASS * 9.81 * x_2) / (x_1 + x_2)) / 100
    N_2 = ((MASS * 9.81 * x_1) / (x_1 + x_2)) / 100

    return (N_1, N_2)
  
def setup_kalman():
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
  # represents uncertainty in measurements
  kf.R *= 70

  # Initialize process noise covariance matrix 
  # represents the uncertainty in the system dynamics
  kf.Q = np.array([[0.01, 0,    0,    0],
                 [0,    0.01, 0,    0],
                 [0,    0,    0.03, 0],
                 [0,    0,    0,    0.03]])
  
  kf.x = np.array([0.5, 0.75, 0., 0.])  # initial state (x, y, vx, vy)
  kf.P = np.eye(4)                   # initial uncertainty

  return kf


# Create an ImageSegmenter object
segment_base_options = python.BaseOptions(model_asset_path='deeplab_v3.tflite')
segment_options = vision.ImageSegmenterOptions(base_options=segment_base_options,
                                       output_category_mask=True)

segmenter = vision.ImageSegmenter.create_from_options(segment_options)

# Create a PoseLandmarker object
detector_base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
detector_options = vision.PoseLandmarkerOptions(
    base_options=detector_base_options)

detector = vision.PoseLandmarker.create_from_options(detector_options)
  
# Init COG smoother class with defined alpha val
alpha = 0.9
data_smoother = SmoothCOG(alpha)

patient_was_unbalanced = False

max_ratio_difference = 0

# setup two kalman filters (one for each foot) in order to smooth joint tracking
# prevents false positives due to jittery predictions from pose estimation model
kf_r = setup_kalman()
kf_l = setup_kalman()

frame_counter = 0

MASS = int(input("Enter your weight in kilograms (kg): "))

while cv2.waitKey(1) < 0:
  ret, frame = vid.read()
  #ret2, frame2 = vid2.read()
  if not ret:
    break

  frame_counter += 1
  
  kernel = (5, 5) # for use in gaussian blur
  blurred_frame = cv2.GaussianBlur(frame, kernel, 0)

  rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=blurred_frame) # convert video frame to mp.Image type 
                                                                     # for processing in pose detector
  ###
  segmentation_result = segmenter.segment(rgb_frame)
  category_mask = segmentation_result.category_mask

  # # Generate solid color images for showing the output segmentation mask.
  # image_data = rgb_frame.numpy_view()
  # fg_image = np.zeros(image_data.shape, dtype=np.uint8)
  # fg_image[:] = (255, 255, 255) # white
  # bg_image = np.zeros(image_data.shape, dtype=np.uint8)
  # bg_image[:] = (192, 192, 192) # gray

  # condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
  # output_image = np.where(condition, fg_image, bg_image)

  # Convert the BGR image to RGB
  image_data = rgb_frame.numpy_view()

  # Apply effects
  black_image = image_data * 0

  # assigns value to pixel depending on if it is part of human body or not
  condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
  output_image = np.where(condition, image_data, black_image)

  cv2.namedWindow("Output3", cv2.WINDOW_NORMAL)
  cv2.imshow("Output3", output_image)
  ###
  
  #rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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
    smoothed_cog = computed_cog

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

    left_foot, right_foot = joint_locations[0], joint_locations[1]

    kf_r.predict()
    kf_l.predict()

    kf_r.update(right_foot)
    kf_l.update(left_foot)

    smoothed_pos_right = (kf_r.x[0], kf_r.x[1])
    smoothed_pos_left = (kf_l.x[0], kf_l.x[1])

    right_ratio, left_ratio = calculate_weight_distribution(rgb_frame.numpy_view(), smoothed_cog, smoothed_pos_right, smoothed_pos_left)

    print(f"Weight distribution difference is: {round(abs(right_ratio - left_ratio), 2)}%")

    if abs(right_ratio - left_ratio) > max_ratio_difference:
      max_ratio_difference = abs(right_ratio - left_ratio)

    # check if subject is "unbalanced" (ignore first 10 frames because of kalman filter initialisation)
    if abs(right_ratio - left_ratio) > 6 and frame_counter >= 10:
        print("###UNBALANCED###")
        patient_was_unbalanced = True
        unbalanced = True
    else:
      unbalanced_frame_counter = 0
      unbalanced = False

    #annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result, computed_cog, smoothed_cog, unbalanced, right_foot, left_foot)
    annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result, smoothed_cog, unbalanced, right_foot, left_foot, smoothed_pos_right, smoothed_pos_left)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Output2", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", annotated_image)
    #cv2.imshow("Output2", frame2)
    #print(f"FRAME: {frame_counter}")

vid.release()
cv2.destroyAllWindows()

print(f"Max ratio difference: {max_ratio_difference}")

print(f"Diagnosis: {patient_was_unbalanced}")
