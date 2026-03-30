import torch
import h5py
from tqdm import tqdm
import torch.nn.functional as F
import dataloader.test_loader as loader


@torch.no_grad()
def extract_DINO_features(model, data_dir, dataset, gnd_fn, split, save_path):
    test_loader = loader.construct_loader("DINOv2", data_dir, dataset, gnd_fn, split)
    num_images = len(test_loader.dataset)

    print(f"Extracting DINO features for {split}...")

    # Open HDF5 file to stream data directly to disk
    with h5py.File(save_path, 'w') as f:
        dset = None
        ptr = 0

        # Added mininterval=30.0 to stop the log spam!
        for batch in tqdm(test_loader, mininterval=30.0):
            if isinstance(batch, (list, tuple)):
                im = batch[0]
            else:
                im = batch

            im = im.to(device='cuda')

            # --- TIMM EXTRACTION LOGIC ---
            features = model.forward_features(im)
            patches = features[:, 1:, :].permute(0, 2, 1)  # (Batch, Dim, Patches)

            # Normalize and move to CPU
            patches = F.normalize(patches, p=2, dim=1).cpu().numpy()
            batch_size = patches.shape[0]

            # Initialize the HDF5 dataset dynamically on the first batch
            if dset is None:
                _, dim, n_patches = patches.shape
                dset = f.create_dataset('features',
                                        shape=(num_images, dim, n_patches),
                                        dtype='float32')

            # Write batch to disk and advance pointer
            dset[ptr:ptr + batch_size] = patches
            ptr += batch_size

    print(f"Finished streaming {split} features to disk.")