from transformers import CLIPProcessor, CLIPModel
from config import cfg as c
from model.CLIP.utils.CLIP_test_model import test_CLIP

def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    model_id = c.CLIP.WEIGHTS

    print(f"Loading CLIP ({model_id}) via HuggingFace...")

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    # Pass model_id as the final argument
    ranks, map_score = test_CLIP(model, processor, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET,
                      c.TEST.CUSTOM, c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.TEST.EVALUATE, model_id)
    return ranks, map_score