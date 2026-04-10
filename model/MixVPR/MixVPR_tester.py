import torch
from config import cfg as c
from model.MixVPR.utils.MixVPR_test_model import test_MixVPR
from model.MixVPR.utils.VPRModel import VPRModel


def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    print("Loading Official ResNet50 + MixVPR...")

    # Official initialization from the WACV 2023 Paper
    model = VPRModel(backbone_arch='resnet50',
                     layers_to_crop=[4],
                     agg_arch='MixVPR',
                     agg_config={'in_channels': 1024,
                                 'in_h': 20,
                                 'in_w': 20,
                                 'out_channels': 1024,
                                 'mix_depth': 4,
                                 'mlp_ratio': 1,
                                 'out_rows': 4},
                     )

    # Load the official PyTorch Lightning checkpoint
    ckpt_path = './weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'

    # Load the full Lightning wrapper into RAM
    lightning_ckpt = torch.load(ckpt_path, map_location='cpu')

    # Extract the weights dictionary and load it into our cleaned model
    model.load_state_dict(lightning_ckpt)

    model = model.to(device)
    model.eval()

    ranks, map_score = test_MixVPR(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET,
                                   c.TEST.CUSTOM, c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.TEST.EVALUATE)
    return ranks, map_score