import os
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model.DINOv2.utils.DINO_utils import extract_DINO_features
from model.SuperGlobal.utils.SG_utils import test_revisitop


@torch.no_grad()
def test_DINO(model, device, cfg, gnd, data_dir, dataset, res, custom, update_data, update_queries, top_m_rerank, evaluate,
              model_id):
    torch.backends.cudnn.benchmark = True
    model.eval()

    print(f'>> {dataset}: Image Retrieval with DINOv2 ({model_id})')
    safe_model_name = str(model_id).replace('/', '_').replace('\\', '_')

    # Use .h5 instead of .pt
    Q_path = os.path.join(data_dir, dataset, f"DINO_query_{safe_model_name}_{res}.h5")
    X_path = os.path.join(data_dir, dataset, f"DINO_data_{safe_model_name}_{res}.h5")

    if update_queries or not os.path.isfile(Q_path):
        extract_DINO_features(model, data_dir, dataset, gnd, "query", Q_path)

    if update_data or not os.path.isfile(X_path):
        extract_DINO_features(model, data_dir, dataset, gnd, "db", X_path)

    # --- LOAD EVERYTHING INTO RAM FOR SPEED ---
    print(">> Loading features...")
    with h5py.File(Q_path, 'r') as f_q:
        Q = f_q['features'][:]
        print(f"Query Shape: {Q.shape}")
        Q_tensor = torch.from_numpy(Q).to(device)

        Q_global = torch.mean(Q_tensor, dim=2)
        Q_global = F.normalize(Q_global, p=2, dim=1)

    with h5py.File(X_path, 'r') as f_x:
        X = f_x['features'][:]
        print(f"Database Shape: {X.shape}")
        X_tensor = torch.from_numpy(X)

    # ---------------------------------------------------------
    # STAGE 1: GLOBAL SEARCH
    # ---------------------------------------------------------
    print(">> Stage 1: Global Descriptor Search...")

    # Compute Database Global Descriptors directly from the RAM tensor
    X_global = F.normalize(torch.mean(X_tensor, dim=2), p=2, dim=1).to(device)

    # Global Similarity
    sim_global = torch.mm(Q_global, X_global.t())
    top_global_scores, top_global_indices = torch.topk(sim_global, k=top_m_rerank, dim=1)

    # ---------------------------------------------------------
    # STAGE 2: LOCAL RERANKING
    # ---------------------------------------------------------
    print(f">> Stage 2: Reranking Top-{top_m_rerank} candidates with Patch Logic...")

    final_ranks = []
    N_q = Q.shape[0]

    for i in tqdm(range(N_q), desc="Reranking", mininterval=5.0):
        q_patches = Q_tensor[i].t().unsqueeze(0)  # (1, 256, 768)
        candidate_idxs = top_global_indices[i].cpu()

        # INSTANT RAM SLICING (No more hard drive bottlenecks)
        db_candidates = X_tensor[candidate_idxs].to(device)

        # Patched Math
        sim_matrix = torch.matmul(q_patches, db_candidates)
        best_match_per_patch, _ = sim_matrix.max(dim=2)

        m_patches = int(Q.shape[2] * 0.5)
        top_m_vals, _ = torch.topk(best_match_per_patch, m_patches, dim=1)
        local_scores = top_m_vals.mean(dim=1)

        local_sort_order = torch.argsort(local_scores, descending=True)
        final_top_m_indices = candidate_idxs[local_sort_order.cpu()]

        global_sort_order = torch.argsort(sim_global[i], descending=True).cpu()
        rest_indices = global_sort_order[top_m_rerank:]

        full_rank_list = torch.cat([final_top_m_indices, rest_indices])
        final_ranks.append(full_rank_list.numpy())

    ranks = np.array(final_ranks).T
    map_score = 0.0

    if evaluate:
        ks = [10, 25, 100]
        if not custom:
            (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

    return ranks, map_score