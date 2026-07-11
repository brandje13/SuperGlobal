import timm
import gc
import torch
from config import cfg as c
from model.DINOv2.utils.DINO_test_model import test_DINO


def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    model_id = c.DINO.WEIGHTS

    print(f"Loading DINOv2 ({model_id}) via timm...")

    # Load DINOv2 model
    model = timm.create_model(model_id, pretrained=True, img_size=c.DINO.RESOLUTION)

    model = model.cuda(device=device)
    model.eval()

    # Execute testing logic
    ranks, map_score = test_DINO(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET, c.DINO.RESOLUTION,
                                 c.TEST.CUSTOM,
                                 c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.DINO.TOP_M, c.TEST.EVALUATE, model_id)

    # Explicitly remove model from VRAM before returning to outer grid search loop
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return ranks, map_score