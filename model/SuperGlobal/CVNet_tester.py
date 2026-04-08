r""" Test code of Correlation Verification Network """
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

def setup_model(device, encoder):
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(c.MODEL.DEPTH, c.MODEL.HEADS.REDUCTION_DIM, c.SupG.relup, encoder)
    model = model.cuda(device=device)
    return model

def __main__(gnd, cfg):
    if c.TEST.WEIGHTS == "":
        print("no utils weights exist!!")
        ranks = []
        map_score = 0.0
    else:
        encoder = ["", ""]
        device = c.MODEL.DEVICE
        model = setup_model(device, encoder)
        checkpoint.load_checkpoint(c.TEST.WEIGHTS, model)

        ranks, map_score = test_model(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET, c.SupG.SCALE_LIST, c.TEST.CUSTOM,
                   c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.SupG.TOP_M, c.SupG.rerank, c.SupG.gemp, c.SupG.rgem,
                   c.SupG.sgem, c.SupG.onemeval, c.MODEL.DEPTH, c.TEST.EVALUATE, logger, c.TEST.WEIGHTS)

    return ranks, map_score