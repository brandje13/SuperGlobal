import os
import numpy as np
import torch

from model.CLIP.utils.CLIP_utils import extract_CLIP_features
from model.SuperGlobal.utils.SG_utils import test_revisitop


@torch.no_grad()
def test_CLIP(model, processor, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, top_k_list):
    torch.backends.cudnn.benchmark = True
    model.eval()

    print(f'>> {dataset}: Pure Text-to-Image Retrieval with CLIP')

    # 1. Load/Extract Text Query Features
    print("extract query features (Text)")
    Q_path = os.path.join(data_dir, dataset, "CLIP_query_features.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_CLIP_features(model, processor, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    # 2. Load/Extract Database Image Features
    print("extract database features (Images)")
    X_path = os.path.join(data_dir, dataset, "CLIP_data_features.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_CLIP_features(model, processor, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query (Text) Shape: {Q.shape}")
    print(f"Database (Image) Shape: {X.shape}")

    # ---------------------------------------------------------
    # GLOBAL SEARCH (Text vs Images)
    # ---------------------------------------------------------
    Q_tensor = torch.from_numpy(Q).to(device)
    X_tensor = torch.from_numpy(X).to(device)

    # Matrix Multiplication of L2-Normalized vectors
    # (N_q, Dim) @ (Dim, N_db) -> (N_q, N_db)
    sim_global = torch.mm(Q_tensor, X_tensor.t())

    # Sort and Transpose to match the RevisitOP evaluation format
    ranks = torch.argsort(sim_global, descending=True).cpu().numpy().T

    # Evaluation
    if True:
        ks = [10, 25, 100]
        if not custom:
            # --- THE FIX: Create a text-specific eval config ---
            cfg_text = cfg.copy()
            cfg_text['gnd'] = cfg['gnd_txt']
            cfg_text['qimlist'] = cfg['qtxtlist']

            # Pass cfg_text instead of cfg
            (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg_text, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

    return ranks