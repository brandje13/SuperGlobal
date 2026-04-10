import os
import torch
import numpy as np
from model.MixVPR.utils.MixVPR_utils import extract_MixVPR_features
from model.SuperGlobal.utils.SG_utils import test_revisitop

@torch.no_grad()
def test_MixVPR(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, evaluate):
    torch.backends.cudnn.benchmark = True

    print(f'>> {dataset}: Pure Global Image Retrieval with ResNet50-MixVPR')

    # Hardcoded 320 resolution tag to prevent overwriting
    Q_path = os.path.join(data_dir, dataset, f"MixVPR_query_resnet50_320.pt")
    X_path = os.path.join(data_dir, dataset, f"MixVPR_data_resnet50_320.pt")

    # 1. Extract Query
    print("extract query features")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_MixVPR_features(model, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    # 2. Extract DB
    print("extract database features")
    if update_data or not os.path.isfile(X_path):
        X = extract_MixVPR_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query Shape: {Q.shape}")
    print(f"Database Shape: {X.shape}")

    # --- GLOBAL SEARCH ---
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