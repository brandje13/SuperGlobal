import os
import numpy as np
import torch

from model.SigLIP.utils.SigLIP_utils import extract_SigLIP_features
from model.SuperGlobal.utils.SG_utils import test_revisitop

@torch.no_grad()
def test_SigLIP(model, processor, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, evaluate, model_id):
    torch.backends.cudnn.benchmark = True
    model.eval()

    print(f'>> {dataset}: Pure Text-to-Image Retrieval with SigLIP 2 ({model_id})')

    # Sanitize the backbone name so it is safe for Windows file paths
    safe_model_name = str(model_id).replace('/', '_').replace('\\', '_')

    # 1. Load/Extract Text Query Features
    print("extract query features (Text)")
    Q_path = os.path.join(data_dir, dataset, f"SigLIP_query_{safe_model_name}.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_SigLIP_features(model, processor, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    # 2. Load/Extract Database Image Features
    print("extract database features (Images)")
    X_path = os.path.join(data_dir, dataset, f"SigLIP_data_{safe_model_name}.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_SigLIP_features(model, processor, data_dir, dataset, gnd, "db")
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

    sim_global = torch.mm(Q_tensor, X_tensor.t())

    ranks = torch.argsort(sim_global, descending=True).cpu().numpy().T
    map_score = 0.0

    if evaluate:
        ks = [10, 25, 100]
        if not custom:
            (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

    return ranks, map_score