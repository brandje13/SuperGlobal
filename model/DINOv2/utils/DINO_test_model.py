import os
import h5py
import numpy as np
import gc
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model.DINOv2.utils.DINO_utils import extract_DINO_features
from model.SuperGlobal.utils.SG_utils import test_revisitop


@torch.no_grad()
def test_DINO(model, device, cfg, gnd, data_dir, dataset, res, custom, update_data, update_queries, top_m_rerank,
              evaluate, model_id):
    torch.backends.cudnn.benchmark = True
    model.eval()

    print(f'>> {dataset}: Image Retrieval with DINOv2 ({model_id})')
    safe_model_name = str(model_id).replace('/', '_').replace('\\', '_')

    Q_path = os.path.join(data_dir, dataset, f"DINO_query_{safe_model_name}_{res}.h5")
    X_path = os.path.join(data_dir, dataset, f"DINO_data_{safe_model_name}_{res}.h5")

    if update_queries or not os.path.isfile(Q_path):
        extract_DINO_features(model, data_dir, dataset, gnd, "query", Q_path)

    if update_data or not os.path.isfile(X_path):
        extract_DINO_features(model, data_dir, dataset, gnd, "db", X_path)

    file_size_gb = os.path.getsize(X_path) / (1024 ** 3)
    USE_RAM_MODE = file_size_gb < 30.0

    print(f">> DB Size: {file_size_gb:.2f} GB | Ultra-Fast RAM Mode: {USE_RAM_MODE}")
    print(">> Loading features and computing Global Descriptors...")

    with h5py.File(Q_path, 'r') as f_q:
        Q = f_q['features'][:]
        # Retain original float16 data type to conserve system RAM
        Q_tensor_cpu = torch.from_numpy(Q)

        # Cast to float32 only for the GPU operations
        Q_global = F.normalize(torch.mean(Q_tensor_cpu.float(), dim=2), p=2, dim=1).to(device)

    # ---------------------------------------------------------
    # STAGE 1: GLOBAL SEARCH
    # ---------------------------------------------------------
    vram_safe_chunk = 250

    if USE_RAM_MODE:
        with h5py.File(X_path, 'r') as f_x:
            # Retain original float16 data type to conserve system RAM
            X_tensor_cpu = torch.from_numpy(f_x['features'][:])
            num_db, dim, n_patches = X_tensor_cpu.shape
            print(f"Database Shape: {X_tensor_cpu.shape}")

        X_global = torch.zeros((num_db, dim), device=device)

        for start_idx in range(0, num_db, vram_safe_chunk):
            end_idx = min(start_idx + vram_safe_chunk, num_db)
            # Cast to float32 when shifting chunk to GPU
            chunk_patches = X_tensor_cpu[start_idx:end_idx].float().to(device)
            X_global[start_idx:end_idx] = torch.mean(chunk_patches, dim=2)

        X_global = F.normalize(X_global, p=2, dim=1)

    else:
        with h5py.File(X_path, 'r') as f_x:
            num_db, dim, n_patches = f_x['features'].shape
            print(f"Database Shape: ({num_db}, {dim}, {n_patches})")

            X_global = torch.zeros((num_db, dim), device=device)

            for start_idx in range(0, num_db, vram_safe_chunk):
                end_idx = min(start_idx + vram_safe_chunk, num_db)
                chunk_patches = torch.from_numpy(f_x['features'][start_idx:end_idx]).to(device).float()
                X_global[start_idx:end_idx] = torch.mean(chunk_patches, dim=2)

            X_global = F.normalize(X_global, p=2, dim=1)

    print(">> Stage 1: Global Descriptor Search...")
    sim_global = torch.mm(Q_global, X_global.t())

    k_fetch = max(1, top_m_rerank)
    top_global_scores, top_global_indices = torch.topk(sim_global, k=k_fetch, dim=1)

    # Move similarity matrix to CPU to free VRAM
    sim_global = sim_global.cpu()

    if top_m_rerank == 0:
        ranks = torch.argsort(sim_global, descending=True).numpy().T
        map_score = 0.0
        if evaluate:
            ks = [10, 25, 100]
            if not custom:
                (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
                print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

        # Free global tensors before returning
        del Q_global, X_global
        torch.cuda.empty_cache()
        gc.collect()

        return ranks, map_score

    # ---------------------------------------------------------
    # STAGE 2: LOCAL RERANKING (Nested Dynamic Hardware Chunking)
    # ---------------------------------------------------------

    # Free Stage 1 global tensors from VRAM prior to Stage 2 iterations
    del Q_global, X_global
    torch.cuda.empty_cache()
    gc.collect()

    print(f">> Stage 2: Reranking Top-{top_m_rerank} candidates with Patch Logic...")

    final_ranks = []
    N_q = Q.shape[0]
    m_patches = int(n_patches * 0.5)

    bytes_per_image_cpu = dim * n_patches * 4
    bytes_per_image_vram = (dim * n_patches + n_patches * n_patches) * 4

    # Target 12GB System RAM and 4GB GPU VRAM
    TARGET_CPU_BYTES = 50 * 1024 * 1024 * 1024
    TARGET_VRAM_BYTES = 30 * 1024 * 1024 * 1024

    cpu_chunk_size = max(100, int(TARGET_CPU_BYTES / bytes_per_image_cpu))
    vram_chunk_size = max(50, int(TARGET_VRAM_BYTES / bytes_per_image_vram))

    print(f">> Hardware Engine Scaling [Res: {res} | Patches: {n_patches}]")
    print(f">> Batching strategy: CPU Chunk = {cpu_chunk_size}, VRAM Chunk = {vram_chunk_size}")

    if USE_RAM_MODE:
        for i in tqdm(range(N_q), desc="Reranking", mininterval=1.0):
            # Cast query chunk to float32 before sending to GPU
            q_patches = Q_tensor_cpu[i].float().to(device).t().unsqueeze(0)
            candidate_idxs = top_global_indices[i].cpu()

            db_candidates_full = X_tensor_cpu[candidate_idxs]
            local_scores = torch.zeros(top_m_rerank)

            for c_start in range(0, top_m_rerank, vram_chunk_size):
                c_end = min(c_start + vram_chunk_size, top_m_rerank)
                # Cast database chunk to float32 before sending to GPU
                db_chunk = db_candidates_full[c_start:c_end].float().to(device)

                sim_matrix = torch.matmul(q_patches, db_chunk)
                best_match_per_patch, _ = sim_matrix.max(dim=2)
                top_m_vals, _ = torch.topk(best_match_per_patch, m_patches, dim=1)

                local_scores[c_start:c_end] = top_m_vals.mean(dim=1).cpu()

            local_sort_order = torch.argsort(local_scores, descending=True)
            final_top_m_indices = candidate_idxs[local_sort_order]

            global_sort_order = torch.argsort(sim_global[i], descending=True)
            rest_indices = global_sort_order[top_m_rerank:]

            final_ranks.append(torch.cat([final_top_m_indices, rest_indices]).numpy())

    else:
        with h5py.File(X_path, 'r') as f_x:
            db_dataset = f_x['features']

            for i in tqdm(range(N_q), desc="Reranking", mininterval=5.0):
                # Cast query chunk to float32 before sending to GPU
                q_patches = Q_tensor_cpu[i].float().to(device).t().unsqueeze(0)
                candidate_idxs = top_global_indices[i].cpu().numpy()

                local_scores = torch.zeros(top_m_rerank)

                sort_order = np.argsort(candidate_idxs)
                sorted_idxs = candidate_idxs[sort_order]

                for cpu_start in range(0, top_m_rerank, cpu_chunk_size):
                    cpu_end = min(cpu_start + cpu_chunk_size, top_m_rerank)
                    batch_sorted_idxs = sorted_idxs[cpu_start:cpu_end]

                    # Read into CPU as float16 to preserve RAM footprint
                    db_cpu_chunk = torch.from_numpy(db_dataset[batch_sorted_idxs.tolist()])

                    for vram_start in range(0, (cpu_end - cpu_start), vram_chunk_size):
                        vram_end = min(vram_start + vram_chunk_size, (cpu_end - cpu_start))

                        # Cast database chunk to float32 before sending to GPU
                        db_gpu_chunk = db_cpu_chunk[vram_start:vram_end].float().to(device)

                        sim_matrix = torch.matmul(q_patches, db_gpu_chunk)
                        best_match_per_patch, _ = sim_matrix.max(dim=2)
                        top_m_vals, _ = torch.topk(best_match_per_patch, m_patches, dim=1)

                        global_offsets = np.arange(cpu_start + vram_start, cpu_start + vram_end)
                        original_rank_positions = sort_order[global_offsets]
                        local_scores[original_rank_positions] = top_m_vals.mean(dim=1).cpu()

                local_sort_order = torch.argsort(local_scores, descending=True)
                final_top_m_indices = torch.tensor(candidate_idxs)[local_sort_order]

                global_sort_order = torch.argsort(sim_global[i], descending=True)
                rest_indices = global_sort_order[top_m_rerank:]

                final_ranks.append(torch.cat([final_top_m_indices, rest_indices]).numpy())

    ranks = np.array(final_ranks).T
    map_score = 0.0

    if evaluate:
        ks = [10, 25, 100]
        if not custom:
            (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

    return ranks, map_score