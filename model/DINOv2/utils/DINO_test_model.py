import os
import h5py
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
    safe_model_name = str(model_id).replace('/', '_').replace('\\', '_')

    # Use .h5 instead of .pt
    Q_path = os.path.join(data_dir, dataset, f"DINO_query_{safe_model_name}.h5")
    X_path = os.path.join(data_dir, dataset, f"DINO_data_{safe_model_name}.h5")

    print("extract query features")
    if update_queries or not os.path.isfile(Q_path):
        extract_DINO_features(model, data_dir, dataset, gnd, "query", Q_path)

    print("extract database features")
    if update_data or not os.path.isfile(X_path):
        extract_DINO_features(model, data_dir, dataset, gnd, "db", X_path)

    # Load Queries entirely into RAM (They are small enough)
    with h5py.File(Q_path, 'r') as f_q:
        Q = f_q['features'][:]
        print(f"Query Shape: {Q.shape}")
        Q_tensor = torch.from_numpy(Q).to(device)

        Q_global = torch.mean(Q_tensor, dim=2)
        Q_global = F.normalize(Q_global, p=2, dim=1)

    # ---------------------------------------------------------
    # STAGE 1: GLOBAL SEARCH (Chunked to save RAM)
    # ---------------------------------------------------------
    print(">> Stage 1: Global Descriptor Search...")

    # Open DB file in read mode, but DO NOT load into RAM
    f_x = h5py.File(X_path, 'r')
    X_dset = f_x['features']
    num_db = X_dset.shape[0]
    print(f"Database Shape: {X_dset.shape}")

    # Compute Database Global Descriptors in chunks
    chunk_size = 1000
    X_global_list = []

    for i in range(0, num_db, chunk_size):
        end = min(i + chunk_size, num_db)
        # Load just this chunk to RAM/GPU
        X_chunk = torch.from_numpy(X_dset[i:end]).to(device)
        X_g_chunk = F.normalize(torch.mean(X_chunk, dim=2), p=2, dim=1)
        X_global_list.append(X_g_chunk)

    X_global = torch.cat(X_global_list, dim=0)

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

        # h5py requires sorted indices to slice efficiently
        sorted_candidate_idxs, sort_order = torch.sort(candidate_idxs)

        # Fetch the candidates directly from the hard drive
        db_candidates_numpy = X_dset[sorted_candidate_idxs.numpy().tolist()]
        db_candidates_sorted = torch.from_numpy(db_candidates_numpy).to(device)

        # Un-sort to restore the global ranking order
        unsort_order = torch.argsort(sort_order)
        db_candidates = db_candidates_sorted[unsort_order]

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

    f_x.close()  # Close the HDF5 file

    ranks = np.array(final_ranks).T
    map_score = 0.0

    if evaluate:
        ks = [10, 25, 100]
        if not custom:
            (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

    return ranks, map_score