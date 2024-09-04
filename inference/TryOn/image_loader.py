from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw
import torch.utils.data as data
import torch.nn as nn 
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
from numpy import asarray

import random
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from shutil import copyfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import os
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from pre_process.viton_hd import VitonHD

label_to_index = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upperclothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "leftShoe": 9,
    "rightShoe": 10,
    "face": 11,
    "leftLeg": 12,
    "rightLeg": 13,
    "leftArm": 14,
    "rightArm": 15,
    "bag": 16,
    "scarf": 17
}

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return torchvision.transforms.Compose(transform_list)


class TryOnDataLoader:
    def __init__(self, avatar_image_url, cloth_image_url, **args):
        """
        Initialize the dataset with the avatar and cloth image URLs, and use VitonHD to process them.
        Args:
        - avatar_image_url: URL to the avatar image.
        - cloth_image_url: URL to the clothing image.
        - viton_hd_processor: VitonHD class instance for image processing.
        """
        self.avatar_image_url = avatar_image_url
        self.cloth_image_url = cloth_image_url
        self.viton_hd_processor = VitonHD()
        self.args = args

        # Process images using VitonHD
        processed_results = self.viton_hd_processor.process_all(avatar_image_url, cloth_image_url, labels_to_remove=[4, 11, 14, 15])

        # Extract the parsed and masked images
        self.avatar_image_parsed = processed_results['image_parse_v3']
        self.avatar_image_agnostic = processed_results['agnostic_image_parse']
        self.cloth_mask = processed_results['cloth_mask']
        self.keypoints_img = processed_results['keypoints_img']
        self.dense_pose = processed_results['dense_pose']

        # Simulate directory behavior with these processed images
        self.source_img = self.avatar_image_parsed  # Parsed avatar image
        self.ref_img = self.cloth_mask              # Masked cloth image
        
        # Set a dummy length for the dataset
        self.length = 1



    def __getitem__(self, index):
        # Instead of loading images from paths, we will use the preprocessed avatar and cloth images.
        source_img = Image.fromarray(self.avatar_image_parsed).convert("RGB")
        ref_img = Image.fromarray(self.cloth_mask).convert("RGB")
        densepose_img = Image.fromarray(self.dense_pose).convert("RGB")
        
        # Process the source image (avatar) using Segformer
        image_tensor = get_tensor()(source_img)

        # If you still want to perform semantic segmentation on the source image using Segformer, use the already initialized model from VitonHD
        processor = self.viton_hd_processor.processor
        model = self.viton_hd_processor.model

        inputs = processor(images=source_img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample the logits to match the original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=source_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted segmentation map
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        parse_array = pred_seg.numpy().astype(np.uint8)

        # Create the garment mask using the new parse_array
        garment_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        garment_mask_with_arms = (parse_array == 4).astype(np.float32) + \
                                (parse_array == 7).astype(np.float32) + \
                                (parse_array == 14).astype(np.float32) + \
                                (parse_array == 15).astype(np.float32)

        epsilon_randomness = random.uniform(0.001, 0.005)
        randomness_range = random.choice([80, 90, 100])
        kernel_size = random.choice([80, 100, 130, 150])

        # Predict mask GT, inpainting mask to be dilated 
        garment_mask = 1 - garment_mask.astype(np.float32)
        garment_mask[garment_mask < 0.5] = 0
        garment_mask[garment_mask >= 0.5] = 1
        garment_mask_resized = cv2.resize(garment_mask, (384, 512), interpolation=cv2.INTER_NEAREST)

        # Ensure the mask is grayscale if it has an extra channel
        if garment_mask_resized.ndim == 3:
            garment_mask_resized = cv2.cvtColor(garment_mask_resized, cv2.COLOR_RGB2GRAY)

        contours, _ = cv2.findContours(((1 - garment_mask_resized) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = epsilon_randomness * cv2.arcLength(max_contour, closed=True)
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, closed=True)
            randomness = np.random.randint(-randomness_range, randomness_range, approx_contour.shape)
            approx_contour = approx_contour + randomness

            zero_mask = np.zeros((512, 384))
            contours = [approx_contour]

            cv2.drawContours(zero_mask, contours, -1, (255), thickness=cv2.FILLED)

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            garment_mask_inpainting = cv2.morphologyEx(zero_mask, cv2.MORPH_CLOSE, kernel)
            garment_mask_inpainting = garment_mask_inpainting.astype(np.float32) / 255.0
            garment_mask_inpainting[garment_mask_inpainting < 0.5] = 0
            garment_mask_inpainting[garment_mask_inpainting >= 0.5] = 1
            garment_mask_inpainting = garment_mask_resized * (1 - garment_mask_inpainting)
        else:
            garment_mask_inpainting = np.zeros((512, 384))

        garment_mask_GT = cv2.erode(garment_mask_resized, self.kernel_dilate, iterations=3)[None]
        garment_mask_inpainting = cv2.erode(garment_mask_inpainting, self.kernel_dilate, iterations=5)[None]

        garment_mask_GT_tensor = torch.from_numpy(garment_mask_GT)
        garment_mask_inpainting_tensor = torch.from_numpy(garment_mask_inpainting)

        # Generate inpainting bounding box, inpainting mask to be dilated
        garment_mask_with_arms = 1 - garment_mask_with_arms.astype(np.float32)
        garment_mask_with_arms[garment_mask_with_arms < 0.5] = 0
        garment_mask_with_arms[garment_mask_with_arms >= 0.5] = 1
        garment_mask_with_arms_resized = cv2.resize(garment_mask_with_arms, (384, 512), interpolation=cv2.INTER_NEAREST)

        # Ensure grayscale for bounding box mask
        if garment_mask_with_arms_resized.ndim == 3:
            garment_mask_with_arms_resized = cv2.cvtColor(garment_mask_with_arms_resized, cv2.COLOR_RGB2GRAY)

        garment_mask_with_arms_boundingbox = cv2.erode(garment_mask_with_arms_resized, self.kernel_dilate, iterations=5)[None]
        
        if garment_mask_with_arms_boundingbox.ndim == 3:
            _, y, x = np.where(garment_mask_with_arms_boundingbox == 0)
        elif garment_mask_with_arms_boundingbox.ndim == 2:
            y, x = np.where(garment_mask_with_arms_boundingbox == 0)
        else:
            raise ValueError("Unexpected shape for garment_mask_with_arms_boundingbox:", garment_mask_with_arms_boundingbox.shape)

        if x.size > 0 and y.size > 0:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            boundingbox = np.ones_like(garment_mask_with_arms_boundingbox)
            boundingbox[:, y_min:y_max, x_min:x_max] = 0
        else:
            boundingbox = np.zeros_like(garment_mask_with_arms_boundingbox)

        boundingbox_tensor = torch.from_numpy(boundingbox)

        # Limit in the bounding box
        garment_mask_inpainting_tensor = torch.where(
            (garment_mask_inpainting_tensor == 0) & (boundingbox_tensor == 0),
            torch.zeros_like(garment_mask_inpainting_tensor),
            torch.ones_like(garment_mask_inpainting_tensor)
        )

        # Select inpainting mask
        if self.state != "test":
            mask_or_boundingbox = random.random()
            if mask_or_boundingbox < 1 - self.boundingbox_as_inpainting_mask_rate:
                inpainting_mask_tensor = garment_mask_inpainting_tensor
            else:
                inpainting_mask_tensor = boundingbox_tensor
        else:
            inpainting_mask_tensor = boundingbox_tensor

        # Combine tensors for final output
        ref_img_combine_tensor = get_tensor()(ref_img)
        poseimage_tensor = get_tensor()(self.keypoints_img)

        # 768 * 512 GT_image
        GT_image_combined = torch.cat((image_tensor, ref_img_combine_tensor), dim=2)

        # 768 * 512 GT_mask
        GT_mask_combined = torch.cat((garment_mask_GT_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)

        # 768 * 512 inpaint_image
        inpaint_image_combined = torch.cat((image_tensor * inpainting_mask_tensor, ref_img_combine_tensor), dim=2)

        # 768 * 512 inpainting mask
        inpainting_mask_combined = torch.cat((inpainting_mask_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)

        # 768 * 512 posemap
        pose_combined = torch.cat((poseimage_tensor, ref_img_combine_tensor), dim=2)

        # 768 * 512 densepose
        densepose_combined = torch.cat((get_tensor()(densepose_img), ref_img_combine_tensor), dim=2)

        # Image name (You can generate or assign a unique name)
        image_name = f"avatar_{index}.jpg"

        return {
            "image_name": image_name,
            "GT_image": GT_image_combined,
            "GT_mask": GT_mask_combined,
            "inpaint_image": inpaint_image_combined,
            "inpaint_mask": inpainting_mask_combined,
            "posemap": pose_combined,
            "densepose": densepose_combined,
            "ref_list": [ref_img_combine_tensor],
        }
