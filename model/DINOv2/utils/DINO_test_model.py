import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model.DINOv2.utils.DINO_utils import extract_DINO_features
from model.SuperGlobal.utils.SG_utils import test_revisitop


@torch.no_grad()
def test_DINO(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, top_m_rerank, evaluate,
              model_id):
    torch.backends.cudnn.benchmark = True
    model.eval()

    print(f'>> {dataset}: Image Retrieval with DINOv2 ({model_id})')

    # Sanitize the backbone name so it is safe for Windows file paths
    safe_model_name = str(model_id).replace('/', '_').replace('\\', '_')

    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, f"DINO_query_{safe_model_name}.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_DINO_features(model, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    print("extract database features")
    X_path = os.path.join(data_dir, dataset, f"DINO_data_{safe_model_name}.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_DINO_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query Shape: {Q.shape}")
    print(f"Database Shape: {X.shape}")

    # ---------------------------------------------------------
    # STAGE 1: GLOBAL SEARCH (Instant Filter)
    # ---------------------------------------------------------
    print(">> Stage 1: Global Descriptor Search...")

    # Convert to Tensor (Keep X on CPU initially)
    Q_tensor = torch.from_numpy(Q).to(device)
    X_tensor = torch.from_numpy(X)

    # Compute Global Descriptors (Mean Pooling)
    Q_global = torch.mean(Q_tensor, dim=2)
    Q_global = F.normalize(Q_global, p=2, dim=1)

    X_global = F.normalize(torch.mean(X_tensor.float(), dim=2), p=2, dim=1).to(device)

    # Global Similarity: (N_q, 768) @ (768, N_db) -> (N_q, N_db)
    sim_global = torch.mm(Q_global, X_global.t())

    # Get Top m Candidates for Reranking
    top_global_scores, top_global_indices = torch.topk(sim_global, k=top_m_rerank, dim=1)

    # ---------------------------------------------------------
    # STAGE 2: LOCAL RERANKING (Detailed Patch Search)
    # ---------------------------------------------------------
    print(f">> Stage 2: Reranking Top-{top_m_rerank} candidates with Patch Logic...")

    final_ranks = []
    N_q = Q.shape[0]

    for i in tqdm(range(N_q), desc="Reranking"):
        # A. Prepare Query Patches
        q_patches = Q_tensor[i].t().unsqueeze(0)  # (1, 256, 768)

        # B. Get the Top m Candidates for this query
        candidate_idxs = top_global_indices[i].cpu()

        # C. Fetch ONLY those m images from the CPU Database
        db_candidates = X_tensor[candidate_idxs].to(device)

        # D. Batched Matrix Multiplication (Small Batch of m)
        sim_matrix = torch.matmul(q_patches, db_candidates)

        # E. Max-Max Scoring
        best_match_per_patch, _ = sim_matrix.max(dim=2)

        # Top 50% of query patches
        m_patches = int(Q.shape[2] * 0.5)
        top_m_vals, _ = torch.topk(best_match_per_patch, m_patches, dim=1)

        # Mean score -> (m,)
        local_scores = top_m_vals.mean(dim=1)

        # F. Re-Sort the Top m
        local_sort_order = torch.argsort(local_scores, descending=True)
        final_top_m_indices = candidate_idxs[local_sort_order.cpu()]

        # G. Append the rest of the list (Ranks k+1 to End)
        global_sort_order = torch.argsort(sim_global[i], descending=True).cpu()
        rest_indices = global_sort_order[top_m_rerank:]

        # Concatenate: [Best m (Reranked)] + [Rest (Global Order)]
        full_rank_list = torch.cat([final_top_m_indices, rest_indices])

        final_ranks.append(full_rank_list.numpy())

    ranks = np.array(final_ranks).T
    map_score = 0.0

    # Evaluation
    if evaluate:
        ks = [10, 25, 100]
        if not custom:
            (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

    return ranks, map_score