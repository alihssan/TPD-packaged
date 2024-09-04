import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from dense_pose_pred import Predictor  # Replace with the correct import for DensePose
from io import BytesIO
from image_parse import ImageParser
from agnostic_mask import AgnosticMasking

class VitonHD:
    def __init__(self, segformer_model_name="mattmdjaga/segformer_b2_clothes"):
        # Initialize the image processor and the model for human parsing (Segformer)
        self.processor = SegformerImageProcessor.from_pretrained(segformer_model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(segformer_model_name)
        
        # Initialize external tools for parsing and mask generation
        self.image_parser_tool = ImageParser()
        self.agnostic_mask_tool = AgnosticMasking()

        # Initialize MediaPipe for pose, face, and hand landmark detection
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.mp_hands = mp.solutions.hands.Hands()

        # Initialize DensePose predictor for body part segmentation
        self.denspose_pred = Predictor()

    def load_image(self, av_img_url, cl_img_url):
        """
        Loads an avatar image and a clothing image from provided URLs.
        Converts them into NumPy arrays for further processing.
        """
        try:
            # Fetch the first image (avatar) and raise error if request fails
            av_response = requests.get(av_img_url)
            av_response.raise_for_status()

            # Fetch the second image (clothing) and raise error if request fails
            cl_response = requests.get(cl_img_url)
            cl_response.raise_for_status()

            # Convert the response to RGB image using PIL and return NumPy arrays
            av_image = Image.open(BytesIO(av_response.content)).convert("RGB")
            cl_image = Image.open(BytesIO(cl_response.content)).convert("RGB")

            return np.array(av_image), np.array(cl_image)

        except UnidentifiedImageError as e:
            print(f"Error: Cannot identify one of the image files from the provided URLs.")
            raise e
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to retrieve one of the images from the URLs.")
            raise e

    def image_parse(self, avatar_image_url):
        """
        Parses an avatar image to get both regular and agnostic segmentation.
        """
        # Perform regular parsing
        mask, colored_image = self.image_parser_tool.regular_parse_image(avatar_image_url)

        # Perform agnostic parsing (removing labels for upper clothes, face, etc.)
        mask_agnostic, colored_image_agnostic = self.image_parser_tool.agnostic_parse_image(avatar_image_url, labels_to_remove=[4, 11, 14, 15])

        return colored_image, colored_image_agnostic

    def agnostic_mask(self, avatar_image_url, upper_clothes_label=4, face_label=11):
        """
        Generates a combined mask for upper clothes and face from the avatar image.
        Applies the mask to remove upper clothes and face from the image.
        """

        # Load the avatar image
        image = self.agnostic_mask_tool.load_image(avatar_image_url)

        # Generate the mask for upper clothes and face
        upper_clothes_mask = self.agnostic_mask_tool.generate_mask(image, [upper_clothes_label])
        face_mask = self.agnostic_mask_tool.generate_mask(image, [face_label])

        # Combine the upper clothes and face masks
        combined_mask = np.clip(upper_clothes_mask + face_mask, 0, 1)

        # Convert combined mask to an image
        combined_mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8)) 

        background_color = [128, 128, 128]  # Grey background
        image_without_upper_clothes_and_face = self.agnostic_mask_tool.apply_mask(image, combined_mask, background_color=background_color)

        return image_without_upper_clothes_and_face, combined_mask_image

    def cloth_mask(self, cloth_image):
        """
        Creates a binary mask for clothing using OpenCV.
        """
        # Convert clothing image to grayscale
        gray = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary mask
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to refine the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def keypoint_detection(self, image):
      """
      Detects pose, face, and hand keypoints from the given image using MediaPipe.
      Outputs the keypoints in a structured format and returns the image with keypoints drawn on a black background.
      
      Args:
          image (np.ndarray): Input image as a NumPy array.

      Returns:
          tuple: (output keypoints as JSON, image with keypoints and connections drawn on a black background)
      """
      # Initialize MediaPipe Pose, Face, and Hands
      mp_pose = mp.solutions.pose
      mp_face_mesh = mp.solutions.face_mesh
      mp_hands = mp.solutions.hands

      pose = mp_pose.Pose()
      face_mesh = mp_face_mesh.FaceMesh()
      hands = mp_hands.Hands()

      # Convert the image to RGB as MediaPipe requires
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Process the image to find the pose, face, and hand landmarks
      pose_results = pose.process(image_rgb)
      face_results = face_mesh.process(image_rgb)
      hands_results = hands.process(image_rgb)

      # Create a black background image
      height, width, _ = image.shape
      black_image = np.zeros((height, width, 3), dtype=np.uint8)

      # Initialize the dictionary to store keypoints in the desired format
      output = {
          "version": 1.3,
          "people": [{
              "person_id": [-1],
              "pose_keypoints_2d": [],
              "face_keypoints_2d": [],
              "hand_left_keypoints_2d": [],
              "hand_right_keypoints_2d": [],
              "pose_keypoints_3d": [],
              "face_keypoints_3d": [],
              "hand_left_keypoints_3d": [],
              "hand_right_keypoints_3d": []
          }]
      }

      # Pose keypoints 2D and connections
      if pose_results.pose_landmarks:
          for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
              x = int(landmark.x * width)
              y = int(landmark.y * height)
              visibility = landmark.visibility
              output["people"][0]["pose_keypoints_2d"].extend([x, y, visibility])
              cv2.circle(black_image, (x, y), 5, (0, 255, 0), -1)  # Draw keypoints

          # Draw connections between pose landmarks
          for connection in mp_pose.POSE_CONNECTIONS:
              start_idx, end_idx = connection
              start_point = (int(pose_results.pose_landmarks.landmark[start_idx].x * width),
                            int(pose_results.pose_landmarks.landmark[start_idx].y * height))
              end_point = (int(pose_results.pose_landmarks.landmark[end_idx].x * width),
                          int(pose_results.pose_landmarks.landmark[end_idx].y * height))
              cv2.line(black_image, start_point, end_point, (0, 255, 0), 2)

      # Face keypoints 2D and connections
      if face_results.multi_face_landmarks:
          for face_landmarks in face_results.multi_face_landmarks:
              for landmark in face_landmarks.landmark:
                  x = int(landmark.x * width)
                  y = int(landmark.y * height)
                  z = landmark.z
                  output["people"][0]["face_keypoints_2d"].extend([x, y, z])
                  cv2.circle(black_image, (x, y), 2, (255, 0, 0), -1)  # Draw keypoints

              # Draw connections between face landmarks
              for connection in mp_face_mesh.FACEMESH_TESSELATION:
                  start_idx, end_idx = connection
                  start_point = (int(face_landmarks.landmark[start_idx].x * width),
                                int(face_landmarks.landmark[start_idx].y * height))
                  end_point = (int(face_landmarks.landmark[end_idx].x * width),
                              int(face_landmarks.landmark[end_idx].y * height))
                  cv2.line(black_image, start_point, end_point, (255, 0, 0), 1)

      # Left hand keypoints 2D and connections
      if hands_results.multi_hand_landmarks:
          for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
              if i == 0:  # Assuming the first detected hand is the left hand
                  for landmark in hand_landmarks.landmark:
                      x = int(landmark.x * width)
                      y = int(landmark.y * height)
                      z = landmark.z
                      output["people"][0]["hand_left_keypoints_2d"].extend([x, y, z])
                      cv2.circle(black_image, (x, y), 3, (0, 0, 255), -1)  # Draw keypoints

                  # Draw connections between left hand landmarks
                  for connection in mp_hands.HAND_CONNECTIONS:
                      start_idx, end_idx = connection
                      start_point = (int(hand_landmarks.landmark[start_idx].x * width),
                                    int(hand_landmarks.landmark[start_idx].y * height))
                      end_point = (int(hand_landmarks.landmark[end_idx].x * width),
                                  int(hand_landmarks.landmark[end_idx].y * height))
                      cv2.line(black_image, start_point, end_point, (0, 0, 255), 2)

      # Right hand keypoints 2D and connections
      if hands_results.multi_hand_landmarks:
          for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
              if i == 1:  # Assuming the second detected hand is the right hand
                  for landmark in hand_landmarks.landmark:
                      x = int(landmark.x * width)
                      y = int(landmark.y * height)
                      z = landmark.z
                      output["people"][0]["hand_right_keypoints_2d"].extend([x, y, z])
                      cv2.circle(black_image, (x, y), 3, (0, 255, 255), -1)  # Draw keypoints

                  # Draw connections between right hand landmarks
                  for connection in mp_hands.HAND_CONNECTIONS:
                      start_idx, end_idx = connection
                      start_point = (int(hand_landmarks.landmark[start_idx].x * width),
                                    int(hand_landmarks.landmark[start_idx].y * height))
                      end_point = (int(hand_landmarks.landmark[end_idx].x * width),
                                  int(hand_landmarks.landmark[end_idx].y * height))
                      cv2.line(black_image, start_point, end_point, (0, 255, 255), 2)

      return output, black_image

    def dense_pose_estimation(self, image):
        """
        Performs dense pose estimation using the DensePose predictor.
        """
        out_image, out_image_seg = self.denspose_pred.predict(image)
        return out_image_seg

    def process_all(self, av_img_url, cl_img_url, labels_to_remove):
        """
        Main method that processes both avatar and clothing images.
        Runs parsing, agnostic parsing, cloth masking, keypoint detection, and dense pose estimation.
        """
        # Load the avatar and clothing images
        av_image, cl_image = self.load_image(av_img_url, cl_img_url)

        # Run Image parsing
        image_parse_v3, agnostic_image_parse_v3 = self.image_parse(av_img_url)

        # Run agnostic masking
        agnostic_mask, agnostic_mask_v3 = self.agnostic_mask(av_img_url)

        # Generate the cloth mask
        cloth_mask = self.cloth_mask(cl_image)

        # Detect keypoints on the avatar image
        keypoints_json, keypoints_img = self.keypoint_detection(av_image)

        # Perform dense pose estimation
        dense_pose = self.dense_pose_estimation(av_image)
        
        return {
            "image_parse_v3": image_parse_v3,
            "agnostic_image_parse": agnostic_image_parse_v3,
            "agnostic_mask": agnostic_mask,
            "agnostic_mask_v3.2": agnostic_mask_v3,
            "cloth_mask": cloth_mask,
            "keypoints_json": keypoints_json,
            "keypoints_img":keypoints_img,
            "dense_pose": dense_pose
        }
