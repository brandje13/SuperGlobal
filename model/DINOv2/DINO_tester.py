import timm
from config import cfg as c
from model.DINOv2.utils.DINO_test_model import test_DINO

def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    model_id = c.DINO.WEIGHTS

    print(f"Loading DINOv2 ({model_id}) via timm...")

    # Load DINOv2 via timm (Compatible with Python < 3.10)
    model = timm.create_model(model_id, pretrained=True, img_size=224)

    model = model.cuda(device=device)
    model.eval()

    # Pass model_id as the final argument
    ranks, map_score = test_DINO(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET, c.TEST.CUSTOM,
                      c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.DINO.TOP_M, c.TEST.EVALUATE, model_id)
    return ranks, map_score