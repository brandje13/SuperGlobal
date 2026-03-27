#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import torch
from dataloader.dataset import DataSet

# Default data directory (/path/pycls/pycls/datasets/data)
from dataloader.dataset_sam import DataSet_SAM
from dataloader.dataset_sg import DataSet_SG
from dataloader.dataset_dino import DataSet_DINO
from dataloader.dataset_clip import DataSet_CLIP


def _construct_loader(model, _DATA_DIR, dataset_name, fn, split, scale_list, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    if model == "SAM":
        dataset = DataSet_SAM(_DATA_DIR, dataset_name, fn, split)
    elif model == "DINOv2":
        dataset = DataSet_DINO(_DATA_DIR, dataset_name, fn, split)
    elif model == "CLIP":
        dataset = DataSet_CLIP(_DATA_DIR, dataset_name, fn, split)
    elif model == "SuperGlobal":
        dataset = DataSet_SG(_DATA_DIR, dataset_name, fn, split, scale_list)
    else:
        dataset = DataSet(_DATA_DIR, dataset_name, fn, split)

    # Windows string multiprocessing safeguard
    workers = 0 if (model == "CLIP" and split == "query") else 4

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=workers,
        pin_memory=False,
        drop_last=drop_last,
    )
    return loader


def construct_loader(model, _DATA_DIR, dataset_name, fn, split, scale_list=None):
    """Test loader wrapper."""
    return _construct_loader(
        model=model,
        _DATA_DIR=_DATA_DIR,
        dataset_name=dataset_name,
        fn=fn,
        split=split,
        scale_list=scale_list,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )