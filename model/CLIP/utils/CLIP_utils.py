import torch
from tqdm import tqdm
import torch.nn.functional as F
import dataloader.test_loader as loader


@torch.no_grad()
def extract_CLIP_features(model, processor, data_dir, dataset, gnd_fn, split):
    """Unified extractor for both text queries and image databases."""
    feats = []

    # The dataloader handles serving Text vs Images based on the split
    test_loader = loader.construct_loader("CLIP", data_dir, dataset, gnd_fn, split)

    print(f"Extracting CLIP features for {split}...")

    for batch in tqdm(test_loader, miniters=1000, maxinterval=1800.0, ascii=True):
        if split == 'query':
            # Batch is a tuple/list of strings
            batch_text = list(batch)
            inputs = processor(text=batch_text, padding=True, truncation=True, return_tensors="pt").to('cuda')
            features = model.get_text_features(**inputs)

        elif split == 'db':
            # Batch is an image tensor
            im = batch[0] if isinstance(batch, (list, tuple)) else batch
            im = im.to(device='cuda')
            features = model.get_image_features(im)

        feats.append(features.detach().cpu())

    feats = torch.cat(feats, dim=0)

    # L2 Normalization is identical for both text and images
    feats = F.normalize(feats, p=2, dim=1)

    return feats.numpy()