import torch
from tqdm import tqdm
import torch.nn.functional as F
import dataloader.test_loader as loader


@torch.no_grad()
def extract_DINO_features(model, data_dir, dataset, gnd_fn, split):
    im_feats = []

    test_loader = loader.construct_loader("DINOv2", data_dir, dataset, gnd_fn, split)

    print(f"Extracting DINO features for {split}...")

    for batch in tqdm(test_loader):
        if isinstance(batch, (list, tuple)):
            im = batch[0]
        else:
            im = batch

        im = im.to(device='cuda')

        # --- TIMM EXTRACTION LOGIC ---
        # timm returns a tensor: (Batch, N_Tokens, Dim)
        # Token 0 is CLS, Tokens 1..End are Patches
        features = model.forward_features(im)

        # Extract Patches (Skip the first [CLS] token)
        # Shape: (Batch, N_Patches, Dim) -> (1, 1369, 768)
        patches = features[:, 1:, :]

        # Transpose to (Batch, Dim, N_Patches) to match our math format
        # (1, 768, 1369)
        patches = patches.permute(0, 2, 1)

        im_feats.append(patches.detach().cpu())

    im_feats = torch.cat(im_feats, dim=0)
    im_feats = F.normalize(im_feats, p=2, dim=1)

    return im_feats.numpy()
