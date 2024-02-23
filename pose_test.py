from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

img = cv2.imread("man_falling.jpg")

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




def draw_landmarks_on_image(rgb_image, detection_result, computed_cog):
  print(f"COMPUTED COG IS : {computed_cog}")
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
  print(x_pixel_location)
  cv2.circle(annotated_image, (x_pixel_location, y_pixel_location), 10, (255, 0, 0), 2)
  return annotated_image

# computes the weighted centre of gravity of subject + plots this on the image
def compute_cog():
  
  x_centre = 0
  y_centre = 0  

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
    counter = 0
    for i in range(0, len(joint_locations)):
       if joint_locations[i][0] in bodyparts_classifications:
         current_bodypart = bodyparts_classifications[joint_locations[i][0]]
         #print(joint_locations[i][0], joint_locations[i][1])
         # multiply each joint location with the appropriate segmentation-weight
         #x_centre += (weights[current_bodypart] * joint_locations[i][1][0])
         #y_centre += (weights[current_bodypart] * joint_locations[i][1][1])
         counter += 1
         x_centre += joint_locations[i][1][0]
         y_centre += joint_locations[i][1][1]
    
    print(counter)
    x_centre /= 13
    y_centre /= 13
    return (x_centre, y_centre)
    



# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("man_falling.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

pose_landmarks_list = detection_result.pose_landmarks

with open('body_part_locations.txt', 'w') as file:
  file.write("Locations (xyz) grouped by body part\n")
  for i in range(0,len(pose_landmarks_list[0])):
    current_body_part = body_parts[i]
    current_landmark = pose_landmarks_list[0][i]
    current_location = (current_landmark.x, current_landmark.y, current_landmark.z)
    file.write(f"{current_body_part}: {current_location}\n")



computed_cog = compute_cog()

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result, computed_cog)
while cv2.waitKey(1) < 0:
  cv2.imshow('', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#cv2.waitKey(1)


