#!/usr/bin/env python3
import argparse
import os
import sys
from yacs.config import CfgNode as CfgNode

_C = CfgNode()
cfg = _C

_C.MODEL_NAME = ""

# ------------------------------------------------------------------------------------ #
# Restored Global Architecture (Required by CVNet Internals)
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()
_C.MODEL.TYPE = "RESNET"
_C.MODEL.DEVICE = 0
_C.MODEL.DEPTH = 101

_C.MODEL.LOSSES = CfgNode()
_C.MODEL.LOSSES.NAME = "cross_entropy"

_C.MODEL.HEADS = CfgNode()
_C.MODEL.HEADS.NAME = "LinearHead"
_C.MODEL.HEADS.IN_FEAT = 2048
_C.MODEL.HEADS.REDUCTION_DIM = 2048

_C.BN = CfgNode()
_C.BN.EPS = 1e-5
_C.BN.MOM = 0.1
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024
_C.BN.ZERO_INIT_FINAL_GAMMA = False
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ------------------------------------------------------------------------------------ #
# Grid Search / Shared Options
# ------------------------------------------------------------------------------------ #
_C.TEST = CfgNode()
_C.TEST.DATA_DIR = ".\datasets"
_C.TEST.DATASET = "ILIAS"
_C.TEST.CUSTOM = False
_C.TEST.UPDATE_DATA = False
_C.TEST.UPDATE_QUERIES = False
_C.TEST.EVALUATE = False
_C.TEST.TOP_K = 10
_C.TEST.WEIGHTS = ".\weights\CVPR2022_CVNet_R101.pyth"

_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True

_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True

# ------------------------------------------------------------------------------------ #
# Ensemble Models
# ------------------------------------------------------------------------------------ #
_C.SupG = CfgNode()
_C.SupG.TOP_M = 1000
_C.SupG.SCALE_LIST = 3
_C.SupG.gemp = True
_C.SupG.sgem = True
_C.SupG.rgem = True
_C.SupG.relup = True
_C.SupG.rerank = True
_C.SupG.onemeval = False

_C.DINO = CfgNode()
_C.DINO.TOP_M = 0
_C.DINO.WEIGHTS = "vit_giant_patch14_dinov2.lvd142m"
_C.DINO.RESOLUTION = 224

_C.CLIP = CfgNode()
_C.CLIP.TOP_M = -1
_C.CLIP.WEIGHTS = "openai/clip-vit-large-patch14"
_C.CLIP.RESOLUTION = 224

_C.SigLIP = CfgNode()
_C.SigLIP.TOP_M = -1
_C.SigLIP.WEIGHTS = "google/siglip-so400m-patch14-384"
_C.SigLIP.RESOLUTION = 384

_C.ConvNeXtV2 = CfgNode()
_C.ConvNeXtV2.TOP_M = -1
_C.ConvNeXtV2.WEIGHTS = "convnextv2_large"
_C.ConvNeXtV2.RESOLUTION = 224
# ------------------------------------------------------------------------------------ #

_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")

def dump_cfg():
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)

def load_cfg(out_dir, cfg_dest="config.yaml"):
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)

def load_cfg_fom_args(description="Config file options."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_list(args.opts)