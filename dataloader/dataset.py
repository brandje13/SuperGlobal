#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import core.transforms as transforms
import torch.utils.data

import json

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


class DataSet(torch.utils.data.Dataset):
    """Base Dataset class."""

    def __init__(self, data_path, dataset, fn, split):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._data_path = data_path
        self._dataset = dataset
        self._fn = fn
        self._split = split
        self._construct_db()
        print("Initializing dataset object")

    def __del__(self):
        print("Deleting dataset object.")

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        if self._dataset in os.listdir(self._data_path):
            with open(os.path.join(self._data_path, self._dataset, self._fn), 'rb') as fin:
                gnd = json.load(fin)
                if self._split == "query":
                    for i in range(len(gnd["qimlist"])):
                        im_fn = gnd["qimlist"][i]
                        im_path = os.path.join(self._data_path, self._dataset, "queries", im_fn)
                        self._db.append(
                            {"im_path": im_path, "bbox": gnd["gnd"][i]["bbx"]})
                elif self._split == "db":
                    for i in range(len(gnd["imlist"])):
                        im_fn = gnd["imlist"][i]
                        im_path = os.path.join(self._data_path, self._dataset, im_fn)
                        self._db.append({"im_path": im_path})
        else:
            assert ()  # Dataset does not exist

    def _load_img(self, index):
        # Load the image
        try:
            im = cv2.imread(self._db[index]["im_path"])

            if self._split == "query":
                bbx = self._db[index]["bbox"]
                im = im[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
            return im
        except:
            print('error: ', self._db[index]["im_path"])
            return None

    def __getitem__(self, index):
        return self._load_img(index)

    def __len__(self):
        return len(self._db)
