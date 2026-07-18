import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import csv
import gc
import pickle
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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


def load_ckpt(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_ckpt(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# Helper target for multiprocessing
def evaluate_combo_chunk(args):
    """
    Evaluates a chunk of combinations. Keeping it as a separate module-level
    function allows Python's multiprocessing pool to serialize and distribute the load.
    """
    chunk, cfg, global_pool_cached, local_pool_cached, semantic_pool_cached, MODES = args
    results = []

    for (g_key, l_key, sem_bb, k, mode) in chunk:
        g_top = global_pool_cached[(g_key, k)]
        l_top = local_pool_cached[(l_key, k)]
        sem_top = semantic_pool_cached[(sem_bb, k)]

        g_info = global_pool_cached[g_key]
        l_info = local_pool_cached[l_key]
        sem_info = semantic_pool_cached[sem_bb]

        models = [
            [g_info['family'], g_top],
            [l_info['family'], l_top],
            [sem_info['family'], sem_top]
        ]

        merged_res = merge_results(cfg, models, mode)
        m_metrics = evaluate_final(cfg, models, merged_res, mode, silent=True)

        results.append({
            'mode': mode,
            'global_family': g_info['family'],
            'global_bb': g_key[0],
            'global_m': g_key[1],
            'global_map': g_info['mAP'],
            'global_time': g_info['time'],
            'local_family': l_info['family'],
            'local_bb': l_key[0],
            'local_m': l_key[1],
            'local_map': l_info['mAP'],
            'local_time': l_info['time'],
            'sem_family': sem_info['family'],
            'sem_bb': sem_bb,
            'sem_map': sem_info['mAP'],
            'sem_time': sem_info['time'],
            'top_k': k,
            'precision': m_metrics['precision'],
            'recall': m_metrics['recall'],
            'f3': m_metrics['f3'],
            'total_time': g_info['time'] + l_info['time'] + sem_info['time']
        })
    return results


def main():
    config.load_cfg_fom_args("Grid Search for Image Retrieval Ensemble")
    c.NUM_GPUS = 1

    # Set to True to skip all untested models and jump straight to Phase 4 fusion.
    FUSE_ONLY_CACHED = False

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

    # Checkpoint Paths
    sg_ckpt = os.path.join(ckpt_dir, "sg_data.pkl")
    conv_ckpt = os.path.join(ckpt_dir, "convnext_data.pkl")
    mixvpr_ckpt = os.path.join(ckpt_dir, "mixvpr_data.pkl")
    dino_ckpt = os.path.join(ckpt_dir, "dino_data.pkl")
    clip_ckpt = os.path.join(ckpt_dir, "clip_data.pkl")
    siglip_ckpt = os.path.join(ckpt_dir, "siglip_data.pkl")

    # Load existing progress
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

    GLOBAL_M_SEARCH = list(range(0, 1100, 100))
    DINO_M_SEARCH = list(range(0, 11000, 1000))
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
                print(f"Scanning for VRAM leaks after SG {sg_bb}...")
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
            print(f"Scanning for VRAM leaks after ConvNeXt {conv_bb}...")
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
                print(f"Scanning for VRAM leaks after MixVPR {mixvpr_bb}...")
                find_leaking_tensors()

    # ====================================================================================
    # PHASE 2: DINOv2 (Local Slot)
    # ====================================================================================
    for dino_bb, res in DINO_BACKBONES:
        c.DINO.WEIGHTS = dino_bb
        c.DINO.RESOLUTION = res
        dino_key = f"{dino_bb}_{res}"

        for m in DINO_M_SEARCH:
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
                print(f"Scanning for VRAM leaks after DINO {dino_key}...")
                find_leaking_tensors()

    # ====================================================================================
    # PHASE 3A: CLIP (Semantic Slot)
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
            print(f"Scanning for VRAM leaks after CLIP {clip_bb}...")
            find_leaking_tensors()

    # ====================================================================================
    # PHASE 3B: SigLIP (Semantic Slot)
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
            print(f"Scanning for VRAM leaks after SigLIP {siglip_bb}...")
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

    total_combos = len(global_pool) * len(local_pool) * len(semantic_pool) * len(TOP_K_SEARCH) * len(MODES)

    if total_combos == 0:
        print(
            "\n[!] Error: One or more pools are completely empty. Cannot run 3-way fusion without at least one model in each slot (Global, Local, Semantic).")
        return

    print(f"\n{'=' * 60}\nFINAL COMBINATORIAL ANALYSIS ({total_combos} combinations)\n{'=' * 60}")

    # --- OPTIMIZATION: PRE-CACHE TOP-K REPRESENTATIONS IN HOST MEMORY ---
    print(">> Pre-calculating and caching Top-K representations...")
    global_pool_cached = {}
    local_pool_cached = {}
    semantic_pool_cached = {}

    for g_key, g_info in global_pool.items():
        # FIX: Strip the massive 'ranks' array from the metadata so it doesn't get pickled
        global_pool_cached[g_key] = {k: v for k, v in g_info.items() if k != 'ranks'}
        for k in TOP_K_SEARCH:
            global_pool_cached[(g_key, k)] = retrieve_top_k(cfg, g_info['ranks'], k, g_info['family'], True)

    for l_key, l_info in local_pool.items():
        # FIX: Strip 'ranks'
        local_pool_cached[l_key] = {k: v for k, v in l_info.items() if k != 'ranks'}
        for k in TOP_K_SEARCH:
            local_pool_cached[(l_key, k)] = retrieve_top_k(cfg, l_info['ranks'], k, l_info['family'], True)

    for s_key, s_info in semantic_pool.items():
        # FIX: Strip 'ranks'
        semantic_pool_cached[s_key] = {k: v for k, v in s_info.items() if k != 'ranks'}
        for k in TOP_K_SEARCH:
            semantic_pool_cached[(s_key, k)] = retrieve_top_k(cfg, s_info['ranks'], k, s_info['family'], True)

    # FIX: Purge the original massive checkpoint pools from System RAM completely
    del global_pool
    del local_pool
    del semantic_pool
    gc.collect()

    # Prepare combinations to distribute
    combinations = []
    for g_key in global_pool_cached.keys():
        # We must filter out the tuple keys (the top-k caches) to safely iterate over the base models
        if isinstance(g_key, tuple): continue
        for l_key in local_pool_cached.keys():
            if isinstance(l_key, tuple): continue
            for s_key in semantic_pool_cached.keys():
                if isinstance(s_key, tuple): continue
                for k in TOP_K_SEARCH:
                    for mode in MODES:
                        combinations.append((g_key, l_key, s_key, k, mode))

    # --- MULTIPROCESSING EXECUTION ---
    # Retrieve physical core counts on Snellius gcn nodes (36 cores/socket, 72 cores total)
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f">> Dispatching grid search to {num_workers} CPU cores...")

    # Segment combinations into chunks for worker nodes to process
    chunk_size = max(1, len(combinations) // (num_workers * 4))
    chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

    tasks = [(chunk, cfg, global_pool_cached, local_pool_cached, semantic_pool_cached, MODES) for chunk in chunks]

    ensemble_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_combo_chunk, task) for task in tasks]

        # Display progression tracking
        for future in tqdm(futures, desc="Fusing Ensembles (Parallelized)", unit="chunk"):
            ensemble_results.extend(future.result())

    # --- FIND THE BEST PATH TO 20/50 ---
    print(f"\n{'*' * 40}\nCONFIGURATIONS MEETING TARGET (P>=0.20, R>=0.50)\n{'*' * 40}")
    targets_met = [r for r in ensemble_results if r['precision'] >= 0.20 and r['recall'] >= 0.50]

    if targets_met:
        sorted_targets = sorted(targets_met, key=lambda x: x['f3'], reverse=True)
        for res in sorted_targets[:20]:
            g_short = os.path.splitext(os.path.basename(res['global_bb']))[0] \
                if res['global_family'] == 'SuperGlobal' else res['global_bb']
            l_short = res['local_bb']
            s_short = res['sem_bb'].split('/')[-1]

            print(
                f"[{res['mode'].upper()}] K:{res['top_k']} | G:{g_short}({res['global_m']}), L:{l_short}({res['local_m']}), S:{s_short} | "
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


if __name__ == "__main__":
    main()