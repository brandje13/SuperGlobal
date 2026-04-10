import torch
from tqdm import tqdm
import torch.nn.functional as F
import dataloader.test_loader as loader


@torch.no_grad()
def extract_MixVPR_features(model, data_dir, dataset, gnd_fn, split):
    feats = []

    # Calls the MixVPR-specific dataloader (320x320)
    test_loader = loader.construct_loader("MixVPR", data_dir, dataset, gnd_fn, split)

    print(f"Extracting MixVPR features for {split}...")

    for batch in tqdm(test_loader):
        im = batch[0] if isinstance(batch, (list, tuple)) else batch
        im = im.to(device='cuda')

        features = model(im)
        feats.append(features.detach().cpu())

    feats = torch.cat(feats, dim=0)

    # L2 Normalize for cosine similarity
    feats = F.normalize(feats, p=2, dim=1)

    return feats.numpy()