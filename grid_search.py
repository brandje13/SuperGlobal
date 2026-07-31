import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import csv
import gc
import pickle
import tempfile
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import config as config
from config import cfg as c

# --- IMPORT ALL TESTERS ---
import model.SuperGlobal.CVNet_tester as CVNet_tester
from model.DINOv2 import DINO_tester
from model.CLIP import CLIP_tester
from model.SigLIP import SigLIP_tester
from model.ConvNeXtV2 import ConvNeXtV2_tester
from model.MixVPR import MixVPR_tester

from utils.config_gnd import config_gnd
from utils.evaluate_final import evaluate_final
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_top_k
from utils.merge_results import merge_results
from utils.cleanup import print_vram_usage, find_leaking_tensors

# --- GLOBAL CACHE FOR WORKERS ---
WORKER_CACHE = {}


def load_ckpt(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_ckpt(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def init_worker(cache_path):
    global WORKER_CACHE
    torch.set_num_threads(1)
    with open(cache_path, 'rb') as f:
        WORKER_CACHE = pickle.load(f)


def evaluate_combo_chunk(chunk):
    """
    Evaluates combinations using the pre-loaded global WORKER_CACHE.
    Supports 1-way, 2-way, and 3-way dynamic ensembles.
    """
    global WORKER_CACHE
    cfg = WORKER_CACHE['cfg']
    global_pool_cached = WORKER_CACHE['global_pool_cached']
    local_pool_cached = WORKER_CACHE['local_pool_cached']
    semantic_pool_cached = WORKER_CACHE['semantic_pool_cached']

    results = []

    for (g_key, l_key, s_key, k, mode) in chunk:
        models = []
        total_time = 0.0

        # --- GLOBAL SLOT ---
        if g_key is not None:
            g_top = global_pool_cached[(g_key, k)]
            g_info = global_pool_cached[g_key]
            models.append([g_info['family'], g_top])
            g_fam, g_bb_str, g_m = g_info['family'], g_key[0], g_key[1]
            g_map, g_time = g_info['mAP'], g_info['time']
            total_time += g_time
        else:
            g_fam, g_bb_str, g_m, g_map, g_time = 'None', 'None', 0, 0.0, 0.0

        # --- LOCAL SLOT ---
        if l_key is not None:
            l_top = local_pool_cached[(l_key, k)]
            l_info = local_pool_cached[l_key]
            models.append([l_info['family'], l_top])
            l_fam, l_bb_str, l_m = l_info['family'], l_key[0], l_key[1]
            l_map, l_time = l_info['mAP'], l_info['time']
            total_time += l_time
        else:
            l_fam, l_bb_str, l_m, l_map, l_time = 'None', 'None', 0, 0.0, 0.0

        # --- SEMANTIC SLOT ---
        if s_key is not None:
            s_top = semantic_pool_cached[(s_key, k)]
            s_info = semantic_pool_cached[s_key]
            models.append([s_info['family'], s_top])
            s_fam, s_bb_str = s_info['family'], s_key
            s_map, s_time = s_info['mAP'], s_info['time']
            total_time += s_time
        else:
            s_fam, s_bb_str, s_map, s_time = 'None', 'None', 0.0, 0.0

        # --- FUSION EXECUTION ---
        num_models = len(models)
        if num_models == 1:
            merged_res = {query: data['top_k'] for query, data in models[0][1].items()}
        else:
            merged_res = merge_results(cfg, models, mode)

        m_metrics = evaluate_final(cfg, models, merged_res, mode, silent=True)

        results.append({
            'combo_type': f"{num_models}-way",
            'mode': mode,
            'global_family': g_fam,
            'global_bb': g_bb_str,
            'global_m': g_m,
            'global_map': g_map,
            'global_time': g_time,
            'local_family': l_fam,
            'local_bb': l_bb_str,
            'local_m': l_m,
            'local_map': l_map,
            'local_time': l_time,
            'sem_family': s_fam,
            'sem_bb': s_bb_str,
            'sem_map': s_map,
            'sem_time': s_time,
            'top_k': k,
            'precision': m_metrics['precision'],
            'recall': m_metrics['recall'],
            'f3': m_metrics['f3'],
            'total_time': total_time
        })
    return results


def main():
    config.load_cfg_fom_args("Grid Search for Image Retrieval Ensemble")
    c.NUM_GPUS = 1

    FUSE_ONLY_CACHED = True

    # --- 1. SETUP GROUND TRUTH ---
    if c.TEST.DATASET in ['roxford5k', 'rparis6k']:
        gnd = f'gnd_{c.TEST.DATASET}.json'
        create_groundtruth_from_txt(c.TEST.DATA_DIR, c.TEST.DATASET)
    elif not c.TEST.DATASET == "":
        query_paths = [os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "queries", i)
                       for i in os.listdir(os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "queries"))]
        create_groundtruth(query_paths, c.TEST.DATA_DIR, c.TEST.DATASET)
        gnd = f'gnd_{c.TEST.DATASET}.json'
    else:
        assert c.TEST.DATASET

    cfg = config_gnd(c.TEST.DATASET, c.TEST.DATA_DIR, c.TEST.CUSTOM, gnd)

    # --- SETUP CHECKPOINT DIRECTORY ---
    ckpt_dir = os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    sg_ckpt = os.path.join(ckpt_dir, "sg_data.pkl")
    conv_ckpt = os.path.join(ckpt_dir, "convnext_data.pkl")
    mixvpr_ckpt = os.path.join(ckpt_dir, "mixvpr_data.pkl")
    dino_ckpt = os.path.join(ckpt_dir, "dino_data.pkl")
    clip_ckpt = os.path.join(ckpt_dir, "clip_data.pkl")
    siglip_ckpt = os.path.join(ckpt_dir, "siglip_data.pkl")

    sg_data = load_ckpt(sg_ckpt)
    conv_data = load_ckpt(conv_ckpt)
    mixvpr_data = load_ckpt(mixvpr_ckpt)
    dino_data = load_ckpt(dino_ckpt)
    clip_data = load_ckpt(clip_ckpt)
    siglip_data = load_ckpt(siglip_ckpt)

    # --- 2. SEARCH SPACE DEFINITIONS ---
    SG_BACKBONES = [
        os.path.join('weights', 'CVPR2022_CVNet_R50.pyth'),
        os.path.join('weights', 'CVPR2022_CVNet_R101.pyth')
    ]

    DINO_BACKBONES = [
        ('vit_small_patch14_dinov2.lvd142m', 224),
        ('vit_base_patch14_dinov2.lvd142m', 224),
        ('vit_large_patch14_dinov2.lvd142m', 224),
        ('vit_giant_patch14_dinov2.lvd142m', 224),
        ('vit_small_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_base_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_large_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_giant_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_large_patch14_reg4_dinov2.lvd142m', 336),
        ('vit_giant_patch14_reg4_dinov2.lvd142m', 336),
        ('vit_giant_patch14_dinov2.lvd142m', 518),
        ('vit_giant_patch14_reg4_dinov2.lvd142m', 518)
    ]

    CLIP_BACKBONES = [
        ('openai/clip-vit-base-patch32', 224),
        ('openai/clip-vit-base-patch16', 224),
        ('openai/clip-vit-large-patch14', 224),
        ('openai/clip-vit-large-patch14-336', 336),
        ('laion/CLIP-ViT-L-14-laion2B-s32B-b82K', 224),
        ('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 224),
        ('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', 224)
    ]

    SIGLIP_BACKBONES = [
        ('google/siglip-base-patch16-224', 224),
        ('google/siglip-base-patch16-256', 256),
        ('google/siglip-base-patch16-384', 384),
        ('google/siglip-large-patch16-256', 256),
        ('google/siglip-large-patch16-384', 384),
        ('google/siglip-so400m-patch14-224', 224),
        ('google/siglip-so400m-patch14-384', 384)
    ]

    CONVNEXT_BACKBONES = [
        ('convnextv2_atto', 224),
        ('convnextv2_femto', 224),
        ('convnextv2_pico', 224),
        ('convnextv2_nano', 224),
        ('convnextv2_tiny', 224),
        ('convnextv2_base', 224),
        ('convnextv2_large', 224),
        ('convnextv2_huge', 224)
    ]

    GLOBAL_M_SEARCH = list(range(0, 100, 100))
    DINO_M_SEARCH = list(range(0, 1000, 1000))
    TOP_K_SEARCH = [10, 50, 100]

    # ====================================================================================
    # PHASE 1A: SuperGlobal (Global Slot)
    # ====================================================================================
    for sg_bb in SG_BACKBONES:
        c.TEST.WEIGHTS = sg_bb
        c.MODEL.DEPTH = 101 if 'R101' in sg_bb else 50
        for m in GLOBAL_M_SEARCH:
            if (sg_bb, m) in sg_data: continue
            if FUSE_ONLY_CACHED: continue

            c.SupG.TOP_M = m
            start = time.time()

            print_vram_usage(f"Pre-SG: {sg_bb} | M={m}")
            try:
                ranks, mAP = CVNet_tester.__main__(gnd, cfg)
                print_vram_usage(f"Post-SG: {sg_bb} | M={m}")
                sg_data[(sg_bb, m)] = {'family': 'SuperGlobal', 'ranks': ranks, 'mAP': mAP, 'time': time.time() - start}
                save_ckpt(sg_data, sg_ckpt)
            except Exception as e:
                print(f"[!] Error on SG {sg_bb}: {e}")
            finally:
                find_leaking_tensors()

    # ====================================================================================
    # PHASE 1B: ConvNeXt V2 (Global Slot)
    # ====================================================================================
    for conv_bb, res in CONVNEXT_BACKBONES:
        c.ConvNeXtV2.WEIGHTS = conv_bb
        c.ConvNeXtV2.RESOLUTION = res
        if (conv_bb, 0) in conv_data: continue
        if FUSE_ONLY_CACHED: continue

        start = time.time()

        print_vram_usage(f"Pre-ConvNeXt: {conv_bb} | Res={res}")
        try:
            ranks, mAP = ConvNeXtV2_tester.__main__(gnd, cfg)
            print_vram_usage(f"Post-ConvNeXt: {conv_bb} | Res={res}")
            conv_data[(conv_bb, 0)] = {'family': 'ConvNeXtV2', 'ranks': ranks, 'mAP': mAP,
                                       'time': time.time() - start}
            save_ckpt(conv_data, conv_ckpt)
        except Exception as e:
            print(f"[!] Error on ConvNeXt {conv_bb}: {e}")
        finally:
            find_leaking_tensors()

    # ====================================================================================
    # PHASE 1C: MixVPR (Global Slot)
    # ====================================================================================
    mixvpr_bb = 'resnet50_MixVPR'
    if (mixvpr_bb, 0) not in mixvpr_data:
        if not FUSE_ONLY_CACHED:
            start = time.time()

            print_vram_usage(f"Pre-MixVPR: {mixvpr_bb}")
            try:
                ranks, mAP = MixVPR_tester.__main__(gnd, cfg)
                print_vram_usage(f"Post-MixVPR: {mixvpr_bb}")
                mixvpr_data[(mixvpr_bb, 0)] = {'family': 'MixVPR', 'ranks': ranks, 'mAP': mAP,
                                               'time': time.time() - start}
                save_ckpt(mixvpr_data, mixvpr_ckpt)
            except Exception as e:
                print(f"[!] Error on MixVPR {mixvpr_bb}: {e}")
            finally:
                find_leaking_tensors()

    # ====================================================================================
    # PHASE 2A: CLIP (Semantic Slot)
    # ====================================================================================
    for clip_bb, res in CLIP_BACKBONES:
        if clip_bb in clip_data: continue
        if FUSE_ONLY_CACHED: continue
        c.CLIP.WEIGHTS = clip_bb
        c.CLIP.RESOLUTION = res
        start = time.time()

        print_vram_usage(f"Pre-CLIP: {clip_bb} | Res={res}")
        try:
            ranks, mAP = CLIP_tester.__main__(gnd, cfg)
            print_vram_usage(f"Post-CLIP: {clip_bb} | Res={res}")
            clip_data[clip_bb] = {'family': 'CLIP', 'ranks': ranks, 'mAP': mAP, 'time': time.time() - start}
            save_ckpt(clip_data, clip_ckpt)
        except Exception as e:
            print(f"[!] Error on CLIP {clip_bb}: {e}")
        finally:
            find_leaking_tensors()

    # ====================================================================================
    # PHASE 2B: SigLIP (Semantic Slot)
    # ====================================================================================
    for siglip_bb, res in SIGLIP_BACKBONES:
        if siglip_bb in siglip_data: continue
        if FUSE_ONLY_CACHED: continue
        c.SigLIP.WEIGHTS = siglip_bb
        c.SigLIP.RESOLUTION = res
        start = time.time()

        print_vram_usage(f"Pre-SigLIP: {siglip_bb} | Res={res}")
        try:
            ranks, mAP = SigLIP_tester.__main__(gnd, cfg)
            print_vram_usage(f"Post-SigLIP: {siglip_bb} | Res={res}")
            siglip_data[siglip_bb] = {'family': 'SigLIP', 'ranks': ranks, 'mAP': mAP, 'time': time.time() - start}
            save_ckpt(siglip_data, siglip_ckpt)
        except Exception as e:
            print(f"[!] Error on SigLIP {siglip_bb}: {e}")
        finally:
            find_leaking_tensors()

    # ====================================================================================
    # PHASE 3: DINOv2 (Local Slot)
    # ====================================================================================
    for m in DINO_M_SEARCH:
        for dino_bb, res in DINO_BACKBONES:
            c.DINO.WEIGHTS = dino_bb
            c.DINO.RESOLUTION = res
            dino_key = f"{dino_bb}_{res}"

            if (dino_key, m) in dino_data:
                print(f">> Skipping DINO {dino_key} M={m} (Loaded from checkpoint)")
                continue
            if FUSE_ONLY_CACHED: continue

            c.DINO.TOP_M = m
            start = time.time()

            print_vram_usage(f"Pre-DINO: {dino_key} | M={m}")
            try:
                ranks, mAP = DINO_tester.__main__(gnd, cfg)
                print_vram_usage(f"Post-DINO: {dino_key} | M={m}")
                dino_data[(dino_key, m)] = {'family': 'DINOv2', 'ranks': ranks, 'mAP': mAP,
                                            'time': time.time() - start}
                save_ckpt(dino_data, dino_ckpt)
            except Exception as e:
                print(f"[!] Error on DINO {dino_key}: {e}")
            finally:
                find_leaking_tensors()

    # ====================================================================================
    # PHASE 4: OPTIMIZED DYNAMIC SLOT-BASED FUSION
    # ====================================================================================
    MODES = ['union', 'intersection', 'majority']

    global_pool = {}
    global_pool.update(sg_data)
    for k, v in conv_data.items():
        clean_key = (k, 0) if isinstance(k, str) else k
        global_pool[clean_key] = v

    for k, v in mixvpr_data.items():
        clean_key = (k, 0) if isinstance(k, str) else k
        global_pool[clean_key] = v

    local_pool = dino_data
    semantic_pool = {**clip_data, **siglip_data}

    print(">> Pre-calculating Top-K representations and stripping matrix bloat...")
    global_pool_cached = {}
    local_pool_cached = {}
    semantic_pool_cached = {}

    for g_key, g_info in global_pool.items():
        global_pool_cached[g_key] = {k: v for k, v in g_info.items() if k != 'ranks'}
        for k in TOP_K_SEARCH:
            global_pool_cached[(g_key, k)] = retrieve_top_k(cfg, g_info['ranks'], k, g_info['family'], True)

    for l_key, l_info in local_pool.items():
        local_pool_cached[l_key] = {k: v for k, v in l_info.items() if k != 'ranks'}
        for k in TOP_K_SEARCH:
            local_pool_cached[(l_key, k)] = retrieve_top_k(cfg, l_info['ranks'], k, l_info['family'], True)

    for s_key, s_info in semantic_pool.items():
        semantic_pool_cached[s_key] = {k: v for k, v in s_info.items() if k != 'ranks'}
        for k in TOP_K_SEARCH:
            semantic_pool_cached[(s_key, k)] = retrieve_top_k(cfg, s_info['ranks'], k, s_info['family'], True)

    print(">> Writing worker cache to local disk to bypass IPC pipe limits...")
    temp_dir = tempfile.gettempdir()
    cache_path = os.path.join(temp_dir, f"fusion_cache_{int(time.time())}.pkl")

    cache_data = {
        'cfg': cfg,
        'global_pool_cached': global_pool_cached,
        'local_pool_cached': local_pool_cached,
        'semantic_pool_cached': semantic_pool_cached
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    # --- DYNAMIC COMBINATION GENERATOR ---
    g_options = list(global_pool.keys()) + [None]
    l_options = list(local_pool.keys()) + [None]
    s_options = list(semantic_pool.keys()) + [None]

    # Fully clear main process RAM before spawning workers
    del cache_data
    del global_pool, local_pool, semantic_pool, sg_data, conv_data, mixvpr_data, dino_data, clip_data, siglip_data
    gc.collect()

    combinations = []
    for g_key in g_options:
        for l_key in l_options:
            for s_key in s_options:
                # Count how many slots are actively filled
                valid_models = [x for x in [g_key, l_key, s_key] if x is not None]
                num_models = len(valid_models)

                # Skip 0-way configurations (no models)
                if num_models == 0:
                    continue

                for k in TOP_K_SEARCH:
                    if num_models == 1:
                        combinations.append((g_key, l_key, s_key, k, 'single'))
                    elif num_models == 2:
                        for mode in ['union', 'intersection']:
                            combinations.append((g_key, l_key, s_key, k, mode))
                    else:
                        for mode in MODES:
                            combinations.append((g_key, l_key, s_key, k, mode))

    total_combos = len(combinations)
    print(f"\n{'=' * 60}\nFINAL COMBINATORIAL ANALYSIS ({total_combos} combinations)\n{'=' * 60}")

    if total_combos == 0:
         print("[!] Error: No combinations generated. Pools may be empty.")
         return

    system_cores = os.cpu_count() or 4
    num_workers = min(system_cores, 12)

    print(f">> Dispatching grid search to {num_workers} CPU cores...")

    chunk_size = max(1, len(combinations) // (num_workers * 50))
    chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

    ensemble_results = []

    # Use initializer to load the massive cache exactly once per worker
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(cache_path,)) as executor:
        futures = [executor.submit(evaluate_combo_chunk, chunk) for chunk in chunks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Ensembles", unit="chunk"):
            ensemble_results.extend(future.result())

    print(f"\n{'*' * 40}\nCONFIGURATIONS MEETING TARGET (P>=0.20, R>=0.50)\n{'*' * 40}")
    targets_met = [r for r in ensemble_results if r['precision'] >= 0.20 and r['recall'] >= 0.50]

    if targets_met:
        sorted_targets = sorted(targets_met, key=lambda x: x['f3'], reverse=True)
        for res in sorted_targets[:20]:
            g_str = f"{os.path.splitext(os.path.basename(res['global_bb']))[0]}({res['global_m']})" if res['global_family'] != 'None' else "None"
            l_str = f"{res['local_bb']}({res['local_m']})" if res['local_family'] != 'None' else "None"
            s_str = f"{res['sem_bb'].split('/')[-1]}" if res['sem_family'] != 'None' else "None"

            print(
                f"[{res['combo_type']} | {res['mode'].upper()}] K:{res['top_k']} | G:{g_str}, L:{l_str}, S:{s_str} | "
                f"P:{res['precision']:.2%}, R:{res['recall']:.2%}, F3:{res['f3']:.4f} | "
                f"mAPs [G:{res['global_map']:.2f}, L:{res['local_map']:.2f}, S:{res['sem_map']:.2f}] | "
                f"Time:{res['total_time']:.1f}s")
    else:
        print("No configuration met the 20/50 target on this dataset.")

    csv_filename = f"grid_search_{c.TEST.DATASET}_{int(time.time())}.csv"
    print(f"\n>> Exporting all {len(ensemble_results)} combinations to {csv_filename}...")

    keys = ensemble_results[0].keys() if ensemble_results else []
    if keys:
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(ensemble_results)
        print(f">> Export complete! Data saved to {os.path.abspath(csv_filename)}")

    if os.path.exists(cache_path):
        os.remove(cache_path)


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()