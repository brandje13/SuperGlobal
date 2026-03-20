import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model.DINOv2.utils.DINO_utils import extract_DINO_features
from model.SuperGlobal.utils.SG_utils import test_revisitop


@torch.no_grad()
def test_DINO(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, top_k_list):
    torch.backends.cudnn.benchmark = True
    model.eval()

    print(f'>> {dataset}: Image Retrieval with DINOv2')

    # 1. Load Features (Same as before)
    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, "DINO_query_features.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_DINO_features(model, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    print("extract database features")
    X_path = os.path.join(data_dir, dataset, "DINO_data_features.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_DINO_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query Shape: {Q.shape}")
    print(f"Database Shape: {X.shape}")

    # Q and X are numpy arrays: (N, 768, 256)
    # 768 = Dimensions, 256 = Patches (14x14 grid)

    # ---------------------------------------------------------
    # STAGE 1: GLOBAL SEARCH (Instant Filter)
    # ---------------------------------------------------------
    print(">> Stage 1: Global Descriptor Search...")

    # Convert to Tensor (Keep X on CPU initially)
    Q_tensor = torch.from_numpy(Q).to(device)
    X_tensor = torch.from_numpy(X)

    # Compute Global Descriptors (Mean Pooling)
    # Shape: (N, 768, 256) -> Mean dim 2 -> (N, 768)
    Q_global = torch.mean(Q_tensor, dim=2)
    Q_global = F.normalize(Q_global, p=2, dim=1)

    # Process DB in chunks to avoid GPU OOM during mean pooling
    # (Though 30k vectors fits easily, being safe)
    X_global = F.normalize(torch.mean(X_tensor.float(), dim=2), p=2, dim=1).to(device)

    # Global Similarity: (N_q, 768) @ (768, N_db) -> (N_q, N_db)
    sim_global = torch.mm(Q_global, X_global.t())

    # Get Top k Candidates for Reranking
    TOP_K_RERANK = 1000
    top_global_scores, top_global_indices = torch.topk(sim_global, k=TOP_K_RERANK, dim=1)

    # ---------------------------------------------------------
    # STAGE 2: LOCAL RERANKING (Detailed Patch Search)
    # ---------------------------------------------------------
    print(f">> Stage 2: Reranking Top-{TOP_K_RERANK} candidates with Patch Logic...")

    final_ranks = []
    N_q = Q.shape[0]

    for i in tqdm(range(N_q), desc="Reranking"):
        # A. Prepare Query Patches
        # Current Q: (1, 768, 256) -> Transpose for matmul -> (1, 256, 768)
        # Q_tensor[i] is (768, 256)
        q_patches = Q_tensor[i].t().unsqueeze(0)  # (1, 256, 768)

        # B. Get the Top k Candidates for this query
        candidate_idxs = top_global_indices[i].cpu()

        # C. Fetch ONLY those k images from the CPU Database
        # X_tensor is (N_db, 768, 256) -> Slice -> (k, 768, 256)
        db_candidates = X_tensor[candidate_idxs].to(device)

        # D. Batched Matrix Multiplication (Small Batch of k)
        # (1, 256, 768) @ (k, 768, 256) -> (k, 256, 256)
        sim_matrix = torch.matmul(q_patches, db_candidates)

        # E. Max-Max Scoring (Same logic as your original code)
        # Max over DB patches (dim 2) -> (k, 256)
        best_match_per_patch, _ = sim_matrix.max(dim=2)

        # Top 50% of query patches
        k_patches = int(Q.shape[2] * 0.5)
        top_k_vals, _ = torch.topk(best_match_per_patch, k_patches, dim=1)

        # Mean score -> (k,)
        local_scores = top_k_vals.mean(dim=1)

        # F. Re-Sort the Top k
        # Sort descending based on new local scores
        local_sort_order = torch.argsort(local_scores, descending=True)

        # Map back to original Database Indices
        final_top_k_indices = candidate_idxs[local_sort_order.cpu()]

        # G. Append the rest of the list (Ranks k+1 to End)
        # We trust the global order for everything past rank k
        global_sort_order = torch.argsort(sim_global[i], descending=True).cpu()
        rest_indices = global_sort_order[TOP_K_RERANK:]

        # Concatenate: [Best k (Reranked)] + [Rest (Global Order)]
        full_rank_list = torch.cat([final_top_k_indices, rest_indices])

        final_ranks.append(full_rank_list.numpy())

    ranks = np.array(final_ranks).T

    # Evaluation
    if True:
        ks = [10, 25, 100]
        if not custom:
            (map, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map * 100, decimals=2)))

    return ranks