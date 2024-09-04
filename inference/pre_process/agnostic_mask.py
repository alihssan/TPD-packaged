from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image, ImageFilter
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

class AgnosticMasking:
    def __init__(self, model_name="mattmdjaga/segformer_b2_clothes"):
        # Initialize the processor and model
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    def load_image(self, image_path_or_url):
        # Load an image from a file path or URL
        if image_path_or_url.startswith('http'):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
        return image

    def generate_mask(self, image, labels):
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample the logits to the size of the original image
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted segmentation and generate a binary mask for the specified labels
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        mask = np.isin(pred_seg, labels).astype(np.uint8)
        return mask

    def dilate_mask(self, mask, size=15):
        # Convert the mask to a PIL image for dilation
        mask_pil = Image.fromarray(mask * 255)
        dilated_mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=size))
        dilated_mask = np.array(dilated_mask_pil) // 255
        return dilated_mask

    def apply_mask(self, image, mask, background_color=[128, 128, 128]):
        # Convert the original image to a NumPy array
        image_np = np.array(image)

        # Apply the grey background color to the masked region
        for i in range(3):  # Loop over the 3 color channels (R, G, B)
            image_np[..., i][mask == 1] = background_color[i]

        # Convert back to a PIL image
        masked_image = Image.fromarray(image_np)
        return masked_image

    def save_image(self, image, path):
        # Save the image to the specified path
        image.save(path)

    def display_image(self, image, title="Image"):
        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def display_mask(self, mask, title="Mask"):
        # Display the mask using matplotlib
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()
