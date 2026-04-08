import timm
from config import cfg as c
from model.ConvNeXtV2.utils.ConvNeXtV2_test_model import test_ConvNeXtV2

def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    model_id = c.ConvNeXtV2.WEIGHTS

    print(f"Loading ConvNeXt V2 ({model_id}) via timm...")

    # num_classes=0 forces the model to act purely as a global feature extractor
    model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device)
    model.eval()

    ranks, map_score = test_ConvNeXtV2(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET,
                      c.TEST.CUSTOM, c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.TEST.EVALUATE, model_id)
    return ranks, map_score