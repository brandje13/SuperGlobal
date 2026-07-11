#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import cv2
import torch
from dataloader.dataset import DataSet
#from segment_anything.utils.transforms import ResizeLongestSide


class DataSet_SAM(DataSet):
    """SAM dataset."""

    def __init__(self, data_path, dataset, fn, split):
        super().__init__(data_path, dataset, fn, split)
        self.transform = 1024

    def __getitem__(self, index):
        # Load the image
        im = self._load_img(index)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.transform.apply_image(im)
        im_tensor = torch.as_tensor(im)
        im_tensor = im_tensor.permute(2, 0, 1)

        return im_tensor
