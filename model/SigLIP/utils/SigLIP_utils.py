import torch
from tqdm import tqdm
import torch.nn.functional as F
import dataloader.test_loader as loader

@torch.no_grad()
def extract_SigLIP_features(model, processor, data_dir, dataset, gnd_fn, split):
    feats = []

    # Make sure your dataloader factory (test_loader.py) routes "SigLIP" to DataSet_SigLIP
    test_loader = loader.construct_loader("SigLIP", data_dir, dataset, gnd_fn, split)

    print(f"Extracting SigLIP features for {split}...")

    for batch in tqdm(test_loader, miniters=1000, maxinterval=1800.0, ascii=True):
        if split == 'query':
            batch_text = list(batch)
            # SigLIP tokenizer handles text identically to CLIP here
            inputs = processor(text=batch_text, padding="max_length", truncation=True, return_tensors="pt").to('cuda')
            features = model.get_text_features(**inputs)

        elif split == 'db':
            im = batch[0] if isinstance(batch, (list, tuple)) else batch
            im = im.to(device='cuda')
            # The kwargs "pixel_values" is the standard for AutoModel vision inputs
            features = model.get_image_features(pixel_values=im)

        feats.append(features.detach().cpu())

    feats = torch.cat(feats, dim=0)
    feats = F.normalize(feats, p=2, dim=1)

    return feats.numpy()