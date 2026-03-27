import timm
from config import cfg as c
from model.DINOv2.utils.DINO_test_model import test_DINO


def __main__(gnd, cfg):
    device = c.SupG.MODEL.DEVICE
    model_id = c.DINO.WEIGHTS

    print("Loading DINOv2 (ViT-B/14) via timm...")

    # Load DINOv2 via timm (Compatible with Python < 3.10)
    # 'vit_base_patch14_dinov2' corresponds to the Base model
    model = timm.create_model(model_id, pretrained=True, img_size=224)

    model = model.cuda(device=device)
    model.eval()

    # Run the DINO test function
    ranks = test_DINO(model, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET, c.TEST.CUSTOM,
                      c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.DINO.TOP_M, c.TEST.EVALUATE)
    return ranks