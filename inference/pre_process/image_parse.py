from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import requests
from io import BytesIO
import numpy as np
from matplotlib import colors

class ImageParser:
    def __init__(self, model_name="mattmdjaga/segformer_b2_clothes"):
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

        # Default color map for regular parsing
        self.default_cmap = colors.ListedColormap([
            'black',    # Background or unclassified
            '#FF4500',  # OrangeRed for face
            '#00FF00',  # Green for the first region
            '#4169E1',  # RoyalBlue for the second region
            '#FFD700',  # Gold for upper body
            '#FF69B4',  # HotPink for variation
            '#8B4513',  # SaddleBrown for another region
            '#40E0D0'   # Turquoise for variation
        ])

        # Agnostic color map for agnostic parsing
        self.agnostic_cmap = colors.ListedColormap([
            'black',    # Background or unclassified
            '#FF4500',  # OrangeRed for face
            '#00FF00',  # Green for the first region
            '#4169E1',  # RoyalBlue for the second region
            '#FFD700',  # Gold for upper body
            '#FF69B4',  # HotPink for variation
            '#8B4513',  # SaddleBrown for another region
            '#40E0D0'   # Turquoise for variation
        ])

    def load_image(self, path_or_url):
        """Loads an image from a URL or a local file path."""
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            response = requests.get(path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(path_or_url)
        return image

    def regular_parse_image(self, image_path_or_url):
        """
        Regular parsing method using the default color map.
        - labels_to_remove: list of label indices to remove by setting to 0 (background).
        """
        image = self.load_image(image_path_or_url)
        inputs = self.processor(images=image, return_tensors="pt")

        # Get model predictions
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample the logits to match the original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted segmentation map as a NumPy array
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

      
        # Ensure pred_seg is a single-channel 8-bit image
        pred_seg = pred_seg.astype(np.uint8)

        # Create an RGB image using the default colormap
        colored_image = self.default_cmap(pred_seg)

        # Convert the RGB image from the colormap to uint8 for saving
        colored_image_uint8 = (colored_image[:, :, :3] * 255).astype(np.uint8)

        return pred_seg, colored_image_uint8

    def agnostic_parse_image(self, image_path_or_url, labels_to_remove=None):
        """
        Agnostic parsing method using the agnostic color map.
        - labels_to_remove: list of label indices to remove by setting to 0 (background).
        """
        image = self.load_image(image_path_or_url)
        inputs = self.processor(images=image, return_tensors="pt")

        # Get model predictions
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample the logits to match the original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted segmentation map as a NumPy array
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

        # Remove specified labels, if provided
        if labels_to_remove:
            for label in labels_to_remove:
                pred_seg[pred_seg == label] = 0

        # Ensure pred_seg is a single-channel 8-bit image
        pred_seg = pred_seg.astype(np.uint8)

        # Create an RGB image using the agnostic colormap
        colored_image = self.agnostic_cmap(pred_seg)

        # Convert the RGB image from the colormap to uint8 for saving
        colored_image_uint8 = (colored_image[:, :, :3] * 255).astype(np.uint8)

        return pred_seg, colored_image_uint8

    def save_images(self, mask_image, colored_image, mask_save_path, colored_save_path):
        """Saves the mask and colored image to the specified paths."""
        mask_image_pil = Image.fromarray(mask_image)
        mask_image_pil.save(mask_save_path)

        colored_image_pil = Image.fromarray(colored_image)
        colored_image_pil.save(colored_save_path)

    def display_image(self, image):
        """Displays an image using matplotlib."""
        plt.imshow(image)
        plt.axis('off')  # Hide axis
        plt.show()
