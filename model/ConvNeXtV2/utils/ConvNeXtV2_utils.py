import torch
from tqdm import tqdm
import torch.nn.functional as F
import dataloader.test_loader as loader


@torch.no_grad()
def extract_ConvNeXtV2_features(model, data_dir, dataset, gnd_fn, split):
    feats = []

    test_loader = loader.construct_loader("ConvNeXtV2", data_dir, dataset, gnd_fn, split)

    print(f"Extracting ConvNeXt V2 features for {split}...")

    for batch in tqdm(test_loader, miniters=1000, maxinterval=1800.0, ascii=True):
        im = batch[0] if isinstance(batch, (list, tuple)) else batch
        im = im.to(device='cuda')

        # Forward pass: timm already handled the pooling
        features = model(im)
        feats.append(features.detach().cpu())

    feats = torch.cat(feats, dim=0)

    # L2 Normalization is mandatory for cosine similarity
    feats = F.normalize(feats, p=2, dim=1)

    return feats.numpy()