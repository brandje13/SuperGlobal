from config import cfg as c
#from segment_anything import sam_model_registry
from model.SAM.utils.SAM_test_model import test_SAM


def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    model = ""  #sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
    model = model.cuda(device=device)

    ranks = test_SAM(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET, c.TEST.CUSTOM,
                     c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.TEST.TOPK_LIST)
    return ranks