import pytorch_lightning as pl
import torch
import torch.nn as nn

# We only need their helper to build the architecture
from model.MixVPR.utils import helper


class VPRModel(pl.LightningModule):
    """
    Stripped down VPRModel strictly for Zero-Shot Inference.
    All training logic, loss functions, and dataloaders have been removed.
    """

    def __init__(self,
                 backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 agg_arch='MixVPR',
                 agg_config={}
                 ):
        super().__init__()

        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # ----------------------------------
        # Get ONLY the backbone and the aggregator
        # This keeps the layer names ('self.backbone', 'self.aggregator') identical
        # so the pre-trained weights map perfectly.
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x