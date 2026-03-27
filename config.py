#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode


# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C

# ------------------------------------------------------------------------------------ #
# Global / Shared Options
# ------------------------------------------------------------------------------------ #
_C.MODEL_NAME = ""

_C.TEST = CfgNode()
_C.TEST.DATA_DIR = ".\datasets"
_C.TEST.DATASET = "ILIAS"
_C.TEST.CUSTOM = False
_C.TEST.UPDATE_DATA = False
_C.TEST.UPDATE_QUERIES = False
_C.TEST.EVALUATE = False
_C.TEST.TOP_K = 10

_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True

_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True

# ------------------------------------------------------------------------------------ #
# Ensemble Model 1: SuperGlobal (CVNet)
# ------------------------------------------------------------------------------------ #
_C.SupG = CfgNode()

# SG Retrieval & Evaluation Variables
_C.SupG.TOP_M = 600
_C.SupG.WEIGHTS = ".\weights\CVPR2022_CVNet_R101.pyth"
_C.SupG.SCALE_LIST = 3

# SG Logic
_C.SupG.gemp = True
_C.SupG.sgem = True
_C.SupG.rgem = True
_C.SupG.relup = True
_C.SupG.rerank = True
_C.SupG.onemeval = False

# SG Architecture (Legacy CVNet Backbone)
_C.SupG.MODEL = CfgNode()
_C.SupG.MODEL.TYPE = "RESNET"
_C.SupG.MODEL.DEVICE = 0
_C.SupG.MODEL.DEPTH = 50

_C.SupG.MODEL.LOSSES = CfgNode()
_C.SupG.MODEL.LOSSES.NAME = "cross_entropy"

_C.SupG.MODEL.HEADS = CfgNode()
_C.SupG.MODEL.HEADS.NAME = "LinearHead"
_C.SupG.MODEL.HEADS.IN_FEAT = 2048
_C.SupG.MODEL.HEADS.REDUCTION_DIM = 2048

# SG Batch Norm
_C.SupG.BN = CfgNode()
_C.SupG.BN.EPS = 1e-5
_C.SupG.BN.MOM = 0.1
_C.SupG.BN.USE_PRECISE_STATS = False
_C.SupG.BN.NUM_SAMPLES_PRECISE = 1024
_C.SupG.BN.ZERO_INIT_FINAL_GAMMA = False
_C.SupG.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.SupG.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ------------------------------------------------------------------------------------ #
# Ensemble Model 2: DINOv2
# ------------------------------------------------------------------------------------ #
_C.DINO = CfgNode()
_C.DINO.TOP_M = 1000
_C.DINO.WEIGHTS = "vit_base_patch14_dinov2.lvd142m"

# ------------------------------------------------------------------------------------ #
# Ensemble Model 3: CLIP
# ------------------------------------------------------------------------------------ #
_C.CLIP = CfgNode()
_C.CLIP.TOP_M = -1
_C.CLIP.WEIGHTS = "openai/clip-vit-base-patch32"

# ------------------------------------------------------------------------------------ #
# Deprecated keys
# ------------------------------------------------------------------------------------ #
_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_list(args.opts)