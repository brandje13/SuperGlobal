import os

from model.MixVPR import MixVPR_tester

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import csv
import gc
import pickle
from tqdm import tqdm

import config as config
from config import cfg as c
from tkfilebrowser import askopenfilenames, askopendirname

# --- IMPORT ALL TESTERS ---
import model.SuperGlobal.CVNet_tester as CVNet_tester
from model.DINOv2 import DINO_tester
from model.CLIP import CLIP_tester
from model.SigLIP import SigLIP_tester
from model.ConvNeXtV2 import ConvNeXtV2_tester

from utils.config_gnd import config_gnd
from utils.evaluate_final import evaluate_final
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_top_k
from utils.merge_results import merge_results


# --- CHECKPOINT HELPERS ---
def load_ckpt(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_ckpt(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def main():
    config.load_cfg_fom_args("Grid Search for Image Retrieval Ensemble")
    c.NUM_GPUS = 1

    # ====================================================================================
    # MASTER SWITCH
    # Set to True to skip all untested models and jump straight to Phase 4 fusion.
    # Set to False to resume normal extraction and calculation.
    # ====================================================================================
    FUSE_ONLY_CACHED = True

    # --- 1. SETUP GROUND TRUTH ---
    if c.TEST.CUSTOM:
        query_paths = askopenfilenames()
        data_dir = askopendirname()
        create_groundtruth(query_paths, data_dir, c.TEST.DATASET)
        gnd = 'custom.json'
    elif c.TEST.DATASET in ['roxford5k', 'rparis6k']:
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
        '.\\weights\\CVPR2022_CVNet_R50.pyth',
        '.\\weights\\CVPR2022_CVNet_R101.pyth'
    ]

    DINO_BACKBONES = [
        # --- Standard DINOv2 (224 Baseline) ---
        ('vit_small_patch14_dinov2.lvd142m', 224),
        ('vit_base_patch14_dinov2.lvd142m', 224),
        ('vit_large_patch14_dinov2.lvd142m', 224),
        ('vit_giant_patch14_dinov2.lvd142m', 224),

        # --- DINOv2 with Registers ---
        ('vit_small_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_base_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_large_patch14_reg4_dinov2.lvd142m', 224),
        ('vit_giant_patch14_reg4_dinov2.lvd142m', 224),

        # --- High-Res Extensions ---
        ('vit_large_patch14_reg4_dinov2.lvd142m', 336),
        ('vit_giant_patch14_reg4_dinov2.lvd142m', 336),
        ('vit_giant_patch14_dinov2.lvd142m', 518),
        ('vit_giant_patch14_reg4_dinov2.lvd142m', 518)
    ]

    CLIP_BACKBONES = [
        # --- OpenAI (224 Native) ---
        ('openai/clip-vit-base-patch32', 224),
        ('openai/clip-vit-base-patch16', 224),
        ('openai/clip-vit-large-patch14', 224),
        ('openai/clip-vit-large-patch14-336', 336),

        # --- OpenCLIP (Trained at 224 but scales well) ---
        ('laion/CLIP-ViT-L-14-laion2B-s32B-b82K', 224),
        ('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 224)  # ,
        # ('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', 224)
    ]

    SIGLIP_BACKBONES = [
        # --- Resolution is baked into the string ---
        ('google/siglip-base-patch16-224', 224),
        ('google/siglip-base-patch16-256', 256),
        ('google/siglip-base-patch16-384', 384),
        ('google/siglip-large-patch16-256', 256),
        ('google/siglip-large-patch16-384', 384),
        ('google/siglip-so400m-patch14-224', 224),
        ('google/siglip-so400m-patch14-384', 384)
    ]

    CONVNEXT_BACKBONES = [
        # --- Standard V2 Scaling (all 224 native) ---
        ('convnextv2_atto', 224),
        ('convnextv2_femto', 224),
        ('convnextv2_pico', 224),
        ('convnextv2_nano', 224),
        ('convnextv2_tiny', 224),
        ('convnextv2_base', 224),
        ('convnextv2_large', 224),
        ('convnextv2_huge', 224)
    ]

    # Search parameters
    GLOBAL_M_SEARCH = list(range(0, 1000, 100))
    DINO_M_SEARCH = list(range(0, 11000, 1000))
    #TOP_K_SEARCH = list(range(10, 110, 10))
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
            try:
                ranks, mAP = CVNet_tester.__main__(gnd, cfg)
                sg_data[(sg_bb, m)] = {'family': 'SuperGlobal', 'ranks': ranks, 'mAP': mAP, 'time': time.time() - start}
                save_ckpt(sg_data, sg_ckpt)
            except Exception as e:
                print(f"[!] Error on SG {sg_bb}: {e}")
            finally:
                torch.cuda.empty_cache();
                gc.collect()

    # ====================================================================================
    # PHASE 1B: ConvNeXt V2 (Global Slot)
    # ====================================================================================
    for conv_bb, res in CONVNEXT_BACKBONES:
        c.ConvNeXtV2.WEIGHTS = conv_bb
        c.ConvNeXtV2.RESOLUTION = res
        if (conv_bb, 0) in conv_data: continue
        if FUSE_ONLY_CACHED: continue

        start = time.time()
        try:
            ranks, mAP = ConvNeXtV2_tester.__main__(gnd, cfg)
            conv_data[(conv_bb, 0)] = {'family': 'ConvNeXtV2', 'ranks': ranks, 'mAP': mAP,
                                       'time': time.time() - start}
            save_ckpt(conv_data, conv_ckpt)
        except Exception as e:
            print(f"[!] Error on ConvNeXt {conv_bb}: {e}")
        finally:
            torch.cuda.empty_cache();
            gc.collect()

    # ====================================================================================
    # PHASE 1C: MixVPR (Global Slot)
    # ====================================================================================
    mixvpr_bb = 'resnet50_MixVPR'
    if (mixvpr_bb, 0) not in mixvpr_data:
        if not FUSE_ONLY_CACHED:
            start = time.time()
            try:
                ranks, mAP = MixVPR_tester.__main__(gnd, cfg)
                mixvpr_data[(mixvpr_bb, 0)] = {'family': 'MixVPR', 'ranks': ranks, 'mAP': mAP,
                                               'time': time.time() - start}
                save_ckpt(mixvpr_data, mixvpr_ckpt)
            except Exception as e:
                print(f"[!] Error on MixVPR {mixvpr_bb}: {e}")
            finally:
                torch.cuda.empty_cache();
                gc.collect()

    # ====================================================================================
    # PHASE 2: DINOv2 (Local Slot)
    # ====================================================================================
    for dino_bb, res in DINO_BACKBONES:
        c.DINO.WEIGHTS = dino_bb
        c.DINO.RESOLUTION = res

        # Concatenate the resolution onto the string to make it unique
        dino_key = f"{dino_bb}_{res}"

        for m in DINO_M_SEARCH:
            if (dino_key, m) in dino_data:
                print(f">> Skipping DINO {dino_key} M={m} (Loaded from checkpoint)")
                continue
            if FUSE_ONLY_CACHED: continue

            c.DINO.TOP_M = m
            start = time.time()
            try:
                ranks, mAP = DINO_tester.__main__(gnd, cfg)
                # Save using the concatenated key
                dino_data[(dino_key, m)] = {'family': 'DINOv2', 'ranks': ranks, 'mAP': mAP,
                                            'time': time.time() - start}
                save_ckpt(dino_data, dino_ckpt)
            except Exception as e:
                print(f"[!] Error on DINO {dino_key}: {e}")
            finally:
                torch.cuda.empty_cache();
                gc.collect()

    # ====================================================================================
    # PHASE 3A: CLIP (Semantic Slot)
    # ====================================================================================
    for clip_bb, res in CLIP_BACKBONES:
        if clip_bb in clip_data: continue
        if FUSE_ONLY_CACHED: continue
        c.CLIP.WEIGHTS = clip_bb
        c.CLIP.RESOLUTION = res
        start = time.time()
        try:
            ranks, mAP = CLIP_tester.__main__(gnd, cfg)
            clip_data[clip_bb] = {'family': 'CLIP', 'ranks': ranks, 'mAP': mAP, 'time': time.time() - start}
            save_ckpt(clip_data, clip_ckpt)
        except Exception as e:
            print(f"[!] Error on CLIP {clip_bb}: {e}")
        finally:
            torch.cuda.empty_cache();
            gc.collect()

    # ====================================================================================
    # PHASE 3B: SigLIP (Semantic Slot)
    # ====================================================================================
    for siglip_bb, res in SIGLIP_BACKBONES:
        if siglip_bb in siglip_data: continue
        if FUSE_ONLY_CACHED: continue
        c.SigLIP.WEIGHTS = siglip_bb
        c.SigLIP.RESOLUTION = res
        start = time.time()
        try:
            ranks, mAP = SigLIP_tester.__main__(gnd, cfg)
            siglip_data[siglip_bb] = {'family': 'SigLIP', 'ranks': ranks, 'mAP': mAP, 'time': time.time() - start}
            save_ckpt(siglip_data, siglip_ckpt)
        except Exception as e:
            print(f"[!] Error on SigLIP {siglip_bb}: {e}")
        finally:
            torch.cuda.empty_cache();
            gc.collect()

    # ====================================================================================
    # PHASE 4: DYNAMIC SLOT-BASED FUSION
    # ====================================================================================
    MODES = ['union', 'intersection', 'majority']

    # Pool the slots together
    global_pool = {}
    global_pool.update(sg_data)
    for k, v in conv_data.items():
        clean_key = (k, 0) if isinstance(k, str) else k
        global_pool[clean_key] = v

    for k, v in mixvpr_data.items():
        clean_key = (k, 0) if isinstance(k, str) else k
        global_pool[clean_key] = v

    local_pool = dino_data  # Currently only DINO occupies this slot
    semantic_pool = {**clip_data, **siglip_data}

    total_combos = len(global_pool) * len(local_pool) * len(semantic_pool) * len(TOP_K_SEARCH) * len(MODES)

    if total_combos == 0:
        print(
            "\n[!] Error: One or more pools are completely empty. Cannot run 3-way fusion without at least one model in each slot (Global, Local, Semantic).")
        return

    print(f"\n{'=' * 60}\nFINAL COMBINATORIAL ANALYSIS ({total_combos} combinations)\n{'=' * 60}")

    ensemble_results = []

    with tqdm(total=total_combos, desc="Fusing Ensembles", unit="combo") as pbar:
        # 1. Iterate over Global Slot
        for (global_bb, global_m), global_info in global_pool.items():

            # 2. Iterate over Local Slot
            for (local_bb, local_m), local_info in local_pool.items():

                # 3. Iterate over Semantic Slot
                for sem_bb, sem_info in semantic_pool.items():

                    total_inf_time = global_info['time'] + local_info['time'] + sem_info['time']

                    for k in TOP_K_SEARCH:
                        # Pass the dynamic family name (e.g., 'ConvNeXtV2' or 'SuperGlobal') to retrieve_top_k
                        global_top = retrieve_top_k(cfg, global_info['ranks'], k, global_info['family'], True)
                        local_top = retrieve_top_k(cfg, local_info['ranks'], k, local_info['family'], True)
                        sem_top = retrieve_top_k(cfg, sem_info['ranks'], k, sem_info['family'], True)

                        models = [
                            [global_info['family'], global_top],
                            [local_info['family'], local_top],
                            [sem_info['family'], sem_top]
                        ]

                        for mode in MODES:
                            merged_res = merge_results(cfg, models, mode)
                            m_metrics = evaluate_final(cfg, models, merged_res, mode, silent=True)

                            ensemble_results.append({
                                'mode': mode,
                                'global_family': global_info['family'],
                                'global_bb': global_bb,
                                'global_m': global_m,
                                'global_map': global_info['mAP'],
                                'global_time': global_info['time'],
                                'local_family': local_info['family'],
                                'local_bb': local_bb,
                                'local_m': local_m,
                                'local_map': local_info['mAP'],
                                'local_time': local_info['time'],
                                'sem_family': sem_info['family'],
                                'sem_bb': sem_bb,
                                'sem_map': sem_info['mAP'],
                                'sem_time': sem_info['time'],
                                'top_k': k,
                                'precision': m_metrics['precision'],
                                'recall': m_metrics['recall'],
                                'f3': m_metrics['f3'],
                                'total_time': total_inf_time
                            })
                            pbar.update(1)

        # --- FIND THE BEST PATH TO 20/50 ---
        print(f"\n{'*' * 40}\nCONFIGURATIONS MEETING TARGET (P>=0.20, R>=0.50)\n{'*' * 40}")
        targets_met = [r for r in ensemble_results if r['precision'] >= 0.20 and r['recall'] >= 0.50]

        if targets_met:
            sorted_targets = sorted(targets_met, key=lambda x: x['f3'], reverse=True)
            for res in sorted_targets[:20]:
                # Safe short-names depending on family
                g_short = res['global_bb'].split('\\')[-1].split('.')[0] if res['global_family'] == 'SuperGlobal' else \
                    res['global_bb']
                l_short = res['local_bb']
                s_short = res['sem_bb'].split('/')[-1]

                print(
                    f"[{res['mode'].upper()}] K:{res['top_k']} | G:{g_short}({res['global_m']}), L:{l_short}({res['local_m']}), S:{s_short} | "
                    f"P:{res['precision']:.2%}, R:{res['recall']:.2%}, F3:{res['f3']:.4f} | "
                    f"mAPs [G:{res['global_map']:.2f}, L:{res['local_map']:.2f}, S:{res['sem_map']:.2f}] | "
                    f"Time:{res['total_time']:.1f}s")
        else:
            print("No configuration met the 20/50 target on this dataset.")

    # --- EXPORT LOGIC ---
    # The export logic remains the same, but your CSV headers will now reflect
    # 'global_bb' and 'sem_bb' instead of strictly 'sg_bb' and 'clip_bb'.

    csv_filename = f"grid_search_{c.TEST.DATASET}_{int(time.time())}.csv"
    print(f"\n>> Exporting all {len(ensemble_results)} combinations to {csv_filename}...")

    keys = ensemble_results[0].keys() if ensemble_results else []
    if keys:
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(ensemble_results)
        print(f">> Export complete! Data saved to {os.getcwd()}\\{csv_filename}")


if __name__ == "__main__":
    main()