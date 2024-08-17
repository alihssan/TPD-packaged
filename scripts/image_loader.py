import os
import random
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return torchvision.transforms.Compose(transform_list)

class ImageLoader:
    def __init__(self, source_image_path, segment_map_path, ref_image_path, pose_image_path, densepose_image_path, boundingbox_as_inpainting_mask_rate=0.4):
        self.source_image_path = source_image_path
        self.segment_map_path = segment_map_path
        self.ref_image_path = ref_image_path
        self.pose_image_path = pose_image_path
        self.densepose_image_path = densepose_image_path
        self.boundingbox_as_inpainting_mask_rate = boundingbox_as_inpainting_mask_rate

        self.kernel = np.ones((1, 1), np.uint8)
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process(self):
        source_img = Image.open(self.source_image_path).convert("RGB")
        source_img = source_img.resize((384,512), Image.BILINEAR)
        image_tensor = get_tensor()(source_img)

        segment_map = Image.open(self.segment_map_path)
        segment_map = segment_map.resize((384,512), Image.NEAREST)
        parse_array = np.array(segment_map)

        garment_mask = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)

        garment_mask_with_arms = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32) + \
                    (parse_array == 14).astype(np.float32) + \
                    (parse_array == 15).astype(np.float32)

        epsilon_randomness = random.uniform(0.001, 0.005)
        randomness_range = random.choice([80, 90, 100])
        kernel_size = random.choice([80, 100, 130, 150])

        garment_mask = 1 - garment_mask.astype(np.float32)
        garment_mask[garment_mask < 0.5] = 0
        garment_mask[garment_mask >= 0.5] = 1
        garment_mask_resized = cv2.resize(garment_mask, (384,512), interpolation=cv2.INTER_NEAREST)

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

            kernel = np.ones((kernel_size,kernel_size),np.uint8)
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

        garment_mask_with_arms = 1 - garment_mask_with_arms.astype(np.float32)
        garment_mask_with_arms[garment_mask_with_arms < 0.5] = 0
        garment_mask_with_arms[garment_mask_with_arms >= 0.5] = 1
        garment_mask_with_arms_resized = cv2.resize(garment_mask_with_arms, (384,512), interpolation=cv2.INTER_NEAREST)

        garment_mask_with_arms_boundingbox = cv2.erode(garment_mask_with_arms_resized, self.kernel_dilate, iterations=5)[None]

        _, y, x = np.where(garment_mask_with_arms_boundingbox == 0)
        if x.size > 0 and y.size > 0:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            boundingbox = np.ones_like(garment_mask_with_arms_boundingbox)
            boundingbox[:, y_min:y_max, x_min:x_max] = 0
        else:
            boundingbox = np.zeros_like(garment_mask_with_arms_boundingbox)

        boundingbox_tensor = torch.from_numpy(boundingbox)

        garment_mask_inpainting_tensor = torch.where((garment_mask_inpainting_tensor==0) & (boundingbox_tensor==0), torch.zeros_like(garment_mask_inpainting_tensor), torch.ones_like(garment_mask_inpainting_tensor))

        inpainting_mask_tensor = boundingbox_tensor

        ref_img_combine = Image.open(self.ref_image_path).convert("RGB")
        ref_img_combine = ref_img_combine.resize((384,512), Image.BILINEAR)
        ref_img_combine_tensor = get_tensor()(ref_img_combine)

        pose_img = Image.open(self.pose_image_path).convert("RGB")
        pose_img = pose_img.resize((384,512), Image.BILINEAR)
        poseimage_tensor = get_tensor()(pose_img)

        densepose_img = Image.open(self.densepose_image_path).convert("RGB")
        densepose_img = densepose_img.resize((384,512), Image.BILINEAR)
        denseposeimage_tensor = get_tensor()(densepose_img)

        inpaint_image = image_tensor * inpainting_mask_tensor

        GT_image_combined = torch.cat((image_tensor, ref_img_combine_tensor), dim=2)
        GT_mask_combined = torch.cat((garment_mask_GT_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)
        inpaint_image_combined = torch.cat((inpaint_image, ref_img_combine_tensor), dim=2)
        inpainting_mask_combined = torch.cat((inpainting_mask_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)
        pose_combined = torch.cat((poseimage_tensor, ref_img_combine_tensor), dim=2)
        densepose_combined = torch.cat((denseposeimage_tensor, ref_img_combine_tensor), dim=2)

        return {
            "GT_image": GT_image_combined,
            "GT_mask": GT_mask_combined,
            "inpaint_image": inpaint_image_combined,
            "inpaint_mask": inpainting_mask_combined,
            "posemap": pose_combined,
            "densepose": densepose_combined,
            "ref_list": [ref_img_combine_tensor],
        }
