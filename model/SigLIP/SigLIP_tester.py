from transformers import AutoProcessor, AutoModel
from config import cfg as c
from model.SigLIP.utils.SigLIP_test_model import test_SigLIP

def __main__(gnd, cfg):
    device = c.MODEL.DEVICE
    # Example config: c.SigLIP.WEIGHTS = "google/siglip2-so400m-patch14-384"
    model_id = c.SigLIP.WEIGHTS

    print(f"Loading SigLIP 2 ({model_id}) via HuggingFace...")

    model = AutoModel.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    ranks, map_score = test_SigLIP(model, processor, device, cfg, gnd, c.TEST.DATA_DIR, c.TEST.DATASET,
                      c.TEST.CUSTOM, c.TEST.UPDATE_DATA, c.TEST.UPDATE_QUERIES, c.TEST.EVALUATE, model_id)
    return ranks, map_score