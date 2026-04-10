import numpy as np
import torch
from torchvision import transforms
from dataloader.dataset import DataSet
from PIL import Image

# Standard ImageNet normalization values (RGB order)
_MEAN = [0.485, 0.456, 0.406]
_SD = [0.229, 0.224, 0.225]


class DataSet_MixVPR(DataSet):
    """MixVPR dataset enforcing 320x320 resolution."""

    def __init__(self, data_path, dataset, fn, split):
        super().__init__(data_path, dataset, fn, split)

        # MixVPR explicitly requires 320x320 resolution
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_SD)
        ])

    def __getitem__(self, index):
        # 1. Load the raw BGR image using the parent class method
        im_bgr = self._load_img(index)

        # 2. Convert from BGR (cv2 default) to RGB (PIL/torchvision default)
        im_rgb = im_bgr[:, :, ::-1]

        # 3. Convert to PIL Image for torchvision transforms
        im_pil = Image.fromarray(im_rgb)

        # 4. Apply resizing and normalization
        im_tensor = self.transform(im_pil)

        return im_tensor