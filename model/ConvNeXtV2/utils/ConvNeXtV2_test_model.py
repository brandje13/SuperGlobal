import os
import numpy as np
import torch

from model.ConvNeXtV2.utils.ConvNeXtV2_utils import extract_ConvNeXtV2_features
from model.SuperGlobal.utils.SG_utils import test_revisitop

@torch.no_grad()
def test_ConvNeXtV2(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, evaluate, model_id):
    torch.backends.cudnn.benchmark = True

    print(f'>> {dataset}: Pure Global Image Retrieval with ConvNeXt V2 ({model_id})')

    safe_model_name = str(model_id).replace('/', '_').replace('\\', '_')

    # 1. Load/Extract Query Features
    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, f"ConvNeXtV2_query_{safe_model_name}.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_ConvNeXtV2_features(model, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    # 2. Load/Extract Database Features
    print("extract database features")
    X_path = os.path.join(data_dir, dataset, f"ConvNeXtV2_data_{safe_model_name}.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_ConvNeXtV2_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query Shape: {Q.shape}")
    print(f"Database Shape: {X.shape}")

    # ---------------------------------------------------------
    # GLOBAL SEARCH
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