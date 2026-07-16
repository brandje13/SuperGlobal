import os
import h5py
import numpy as np
import gc
import psutil
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

    file_size_bytes = os.path.getsize(X_path)
    file_size_gb = file_size_bytes / (1024 ** 3)

    cgroup_mem_limit_path = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
    cgroup_mem_usage_path = '/sys/fs/cgroup/memory/memory.usage_in_bytes'

    available_ram = psutil.virtual_memory().available

    if os.path.exists(cgroup_mem_limit_path) and os.path.exists(cgroup_mem_usage_path):
        with open(cgroup_mem_limit_path, 'r') as f_limit, open(cgroup_mem_usage_path, 'r') as f_usage:
            cgroup_limit = int(f_limit.read().strip())
            cgroup_usage = int(f_usage.read().strip())
            if cgroup_limit < (1024 ** 4):
                available_ram = min(available_ram, cgroup_limit - cgroup_usage)

    USE_RAM_MODE = file_size_bytes < (available_ram * 0.60)

    print(f">> DB Size: {file_size_gb:.2f} GB | Ultra-Fast RAM Mode: {USE_RAM_MODE}")
    print(">> Loading features and computing Global Descriptors...")

    with h5py.File(Q_path, 'r') as f_q:
        Q = f_q['features'][:]
        Q_tensor_cpu = torch.from_numpy(Q)

        Q_global = F.normalize(torch.mean(Q_tensor_cpu.float(), dim=2), p=2, dim=1).to(device)

    # ---------------------------------------------------------
    # STAGE 1: GLOBAL SEARCH
    # ---------------------------------------------------------
    vram_safe_chunk = 250

    if USE_RAM_MODE:
        with h5py.File(X_path, 'r') as f_x:
            db_shape = f_x['features'].shape
            np_dtype = f_x['features'].dtype
            pt_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype

            print(f">> Pre-allocating {db_shape} pinned tensor to bypass RAM spikes...")
            X_tensor_cpu = torch.empty(db_shape, dtype=pt_dtype).pin_memory()
            f_x['features'].read_direct(X_tensor_cpu.numpy())

            num_db, dim, n_patches = X_tensor_cpu.shape
            print(f"Database Shape: {X_tensor_cpu.shape}")

        X_global = torch.zeros((num_db, dim), device=device)

        for start_idx in range(0, num_db, vram_safe_chunk):
            end_idx = min(start_idx + vram_safe_chunk, num_db)
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

    sim_global = sim_global.cpu()

    if top_m_rerank == 0:
        ranks = torch.argsort(sim_global, descending=True).numpy().T
        map_score = 0.0
        if evaluate:
            ks = [10, 25, 100]
            if not custom:
                (map_score, _, _, _), (_, _, _, _), (_, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
                print('Retrieval results {}: mAP: {}'.format(dataset, np.around(map_score * 100, decimals=2)))

        del Q_global, X_global
        if 'X_tensor_cpu' in locals(): del X_tensor_cpu
        if 'Q_tensor_cpu' in locals(): del Q_tensor_cpu

        torch.cuda.empty_cache()
        gc.collect()

        return ranks, map_score

    # ---------------------------------------------------------
    # STAGE 2: LOCAL RERANKING (Nested Dynamic Hardware Chunking)
    # ---------------------------------------------------------

    del Q_global, X_global
    torch.cuda.empty_cache()
    gc.collect()

    print(f">> Stage 2: Reranking Top-{top_m_rerank} candidates with Patch Logic...")

    final_ranks = []
    N_q = Q.shape[0]
    m_patches = int(n_patches * 0.5)

    bytes_per_image_cpu = dim * n_patches * 4
    bytes_per_image_vram = (dim * n_patches + n_patches * n_patches) * 4

    total_vram = torch.cuda.get_device_properties(device).total_memory
    allocated_vram = torch.cuda.memory_allocated(device)

    safe_available_vram = (total_vram - allocated_vram) * 0.85
    TARGET_VRAM_BYTES = int(safe_available_vram)
    TARGET_CPU_BYTES = int(available_ram * 0.80)

    print(
        f">> Dynamic Allocations | CPU Target: {TARGET_CPU_BYTES / (1024 ** 3):.2f} GB | VRAM Target: {TARGET_VRAM_BYTES / (1024 ** 3):.2f} GB")

    cpu_chunk_size = min(top_m_rerank, max(100, int(TARGET_CPU_BYTES / bytes_per_image_cpu)))
    vram_chunk_size = min(top_m_rerank, max(50, int(TARGET_VRAM_BYTES / bytes_per_image_vram)))

    print(f">> Hardware Engine Scaling [Res: {res} | Patches: {n_patches}]")
    print(f">> Batching strategy: CPU Chunk = {cpu_chunk_size}, VRAM Chunk = {vram_chunk_size}")

    db_gpu_buffer = torch.zeros((vram_chunk_size, dim, n_patches), dtype=torch.float32, device=device)

    if USE_RAM_MODE:
        for i in tqdm(range(N_q), desc="Reranking", mininterval=1.0):
            q_patches = Q_tensor_cpu[i].float().to(device).t().unsqueeze(0)
            candidate_idxs = top_global_indices[i].cpu()

            # 1. Sort indices to prevent CPU RAM thrashing (Mirroring your Disk-mode logic)
            sort_order = torch.argsort(candidate_idxs)
            sorted_candidate_idxs = candidate_idxs[sort_order]

            local_scores = torch.zeros(top_m_rerank)

            for c_start in range(0, top_m_rerank, vram_chunk_size):
                c_end = min(c_start + vram_chunk_size, top_m_rerank)
                current_batch_size = c_end - c_start

                # Slice from the sequentially sorted array
                chunk_candidate_idxs = sorted_candidate_idxs[c_start:c_end].tolist()

                # 2. Your original Zero-Allocation DMA loop (Now executing straight down the RAM sticks)
                for local_idx, db_idx in enumerate(chunk_candidate_idxs):
                    db_gpu_buffer[local_idx].copy_(X_tensor_cpu[db_idx], non_blocking=True)

                db_chunk = db_gpu_buffer[:current_batch_size]

                sim_matrix = torch.matmul(q_patches, db_chunk)
                best_match_per_patch, _ = sim_matrix.max(dim=2)
                top_m_vals, _ = torch.topk(best_match_per_patch, m_patches, dim=1)

                # 3. Re-map the calculated batch scores back to their original global rank positions
                global_offsets = torch.arange(c_start, c_end)
                original_rank_positions = sort_order[global_offsets]
                local_scores[original_rank_positions] = top_m_vals.mean(dim=1).cpu()

                del sim_matrix, best_match_per_patch, top_m_vals

            local_sort_order = torch.argsort(local_scores, descending=True)
            final_top_m_indices = candidate_idxs[local_sort_order]

            global_sort_order = torch.argsort(sim_global[i], descending=True)
            rest_indices = global_sort_order[top_m_rerank:]

            final_ranks.append(torch.cat([final_top_m_indices, rest_indices]).numpy())
    else:
        with h5py.File(X_path, 'r') as f_x:
            db_dataset = f_x['features']

            for i in tqdm(range(N_q), desc="Reranking", mininterval=5.0):
                q_patches = Q_tensor_cpu[i].float().to(device).t().unsqueeze(0)

                candidate_idxs_tensor = top_global_indices[i].cpu()
                candidate_idxs_np = candidate_idxs_tensor.numpy()

                local_scores = torch.zeros(top_m_rerank)

                sort_order = np.argsort(candidate_idxs_np)
                sorted_idxs = candidate_idxs_np[sort_order]

                for cpu_start in range(0, top_m_rerank, cpu_chunk_size):
                    cpu_end = min(cpu_start + cpu_chunk_size, top_m_rerank)
                    batch_sorted_idxs = sorted_idxs[cpu_start:cpu_end]

                    db_cpu_chunk = torch.from_numpy(db_dataset[batch_sorted_idxs.tolist()]).pin_memory()

                    for vram_start in range(0, (cpu_end - cpu_start), vram_chunk_size):
                        vram_end = min(vram_start + vram_chunk_size, (cpu_end - cpu_start))
                        current_batch_size = vram_end - vram_start

                        db_gpu_buffer[:current_batch_size].copy_(db_cpu_chunk[vram_start:vram_end], non_blocking=True)
                        db_gpu_chunk = db_gpu_buffer[:current_batch_size]

                        sim_matrix = torch.matmul(q_patches, db_gpu_chunk)
                        best_match_per_patch, _ = sim_matrix.max(dim=2)
                        top_m_vals, _ = torch.topk(best_match_per_patch, m_patches, dim=1)

                        global_offsets = np.arange(cpu_start + vram_start, cpu_start + vram_end)
                        original_rank_positions = sort_order[global_offsets]
                        local_scores[original_rank_positions] = top_m_vals.mean(dim=1).cpu()

                        del sim_matrix, best_match_per_patch, top_m_vals

                    del db_cpu_chunk

                local_sort_order = torch.argsort(local_scores, descending=True)
                final_top_m_indices = candidate_idxs_tensor[local_sort_order]

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

    if 'X_tensor_cpu' in locals():
        del X_tensor_cpu
    if 'Q_tensor_cpu' in locals():
        del Q_tensor_cpu
    if 'db_gpu_buffer' in locals():
        del db_gpu_buffer

    gc.collect()
    torch.cuda.empty_cache()

    return ranks, map_score