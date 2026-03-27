r""" Test code of Correlation Verification Network """
# written by Seongwon Lee (won4113@yonsei.ac.kr)

import core.checkpoint as checkpoint
from model.SuperGlobal.CVNet_Rerank_model import CVNet_Rerank
from model.SuperGlobal.utils.SG_test_model import test_model
import logging
from config import cfg as c

logger = logging.getLogger(__name__)

logger.setLevel(level=logging.INFO)

handler = logging.FileHandler("log.txt")

handler.setLevel(logging.INFO)

logger.addHandler(handler)

#logger.info("Start print log")


def setup_model(device, encoder):
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(c.SupG.MODEL.DEPTH, c.SupG.MODEL.HEADS.REDUCTION_DIM, c.SupG.relup, encoder)
    #print(model)
    model = model.cuda(device=device)

    return model


def __main__(gnd, cfg):
    """Test the model."""
    if c.SupG.WEIGHTS == "":
        print("no utils weights exist!!")
        ranks = []
    else:
        # Construct the model
        encoder = ["", ""]
        device = c.SupG.MODEL.DEVICE
        model = setup_model(device, encoder)
        # Load checkpoint
        checkpoint.load_checkpoint(c.SupG.WEIGHTS, model)

        ranks, map_score = test_model(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET, c.SupG.SCALE_LIST, c.TEST.CUSTOM,
                   c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.SupG.TOP_M, c.SupG.rerank, c.SupG.gemp, c.SupG.rgem,
                   c.SupG.sgem, c.SupG.onemeval, c.SupG.MODEL.DEPTH, c.TEST.EVALUATE, logger)

    return ranks, map_score