import os
import time
import torch
import csv
import gc

import config as config
from config import cfg as c
from tkfilebrowser import askopenfilenames, askopendirname

import model.SuperGlobal.CVNet_tester as CVNet_tester
from model.DINOv2 import DINO_tester
from model.CLIP import CLIP_tester

from utils.config_gnd import config_gnd
from utils.evaluate_final import evaluate_final
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_top_k, save_merged_results
from utils.merge_results import merge_results


def main():
    config.load_cfg_fom_args("Grid Search for Image Retrieval Ensemble")
    c.NUM_GPUS = 1
    # We do NOT freeze the config here because we need to mutate TOP_M dynamically
    # c.freeze()

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

    # --- 2. DEFINE SEARCH SPACE ---
    SG_M_SEARCH = list(range(1000, 11000, 1000))  # CVNet Global Reranking limits
    DINO_M_SEARCH = list(range(1000, 11000, 1000))  # DINOv2 Patch Reranking limits
    TOP_K_SEARCH = list(range(10, 160, 10))  # Final mining depths (Zero-Compute)

    sg_data = {}
    dino_data = {}

    # PHASE 1: SuperGlobal Sweep
    for m in SG_M_SEARCH:
        c.SupG.TOP_M = m
        start = time.time()

        try:
            ranks, mAP = CVNet_tester.__main__(gnd, cfg)
            elapsed = time.time() - start
            sg_data[m] = {'ranks': ranks, 'mAP': mAP, 'time': elapsed}

        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] WARNING: CUDA Out of Memory at SuperGlobal M={m}. Skipping configuration.")
        except RuntimeError as e:
            # Fallback catch for older PyTorch versions that treat OOM as a standard RuntimeError
            if "out of memory" in str(e).lower():
                print(f"\n[!] WARNING: CUDA Out of Memory at SuperGlobal M={m}. Skipping configuration.")
            else:
                raise e  # If it's a different bug, we still want it to crash so you can fix it

        finally:
            # SAFETY: Always run this, whether it succeeded or OOM'd
            torch.cuda.empty_cache()
            gc.collect()

    # PHASE 2: DINOv2 Sweep
    for m in DINO_M_SEARCH:
        c.DINO.TOP_M = m
        start = time.time()

        try:
            ranks, mAP = DINO_tester.__main__(gnd, cfg)
            elapsed = time.time() - start
            dino_data[m] = {'ranks': ranks, 'mAP': mAP, 'time': elapsed}

        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] WARNING: CUDA Out of Memory at DINOv2 M={m}. Skipping configuration.")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[!] WARNING: CUDA Out of Memory at DINOv2 M={m}. Skipping configuration.")
            else:
                raise e

        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # PHASE 3: CLIP (Constant baseline)
    start = time.time()
    clip_ranks, clip_mAP = CLIP_tester.__main__(gnd, cfg)
    clip_time = time.time() - start

    # ====================================================================================
    # PHASE 4: THE FULL COMBINATORIAL FRONTIER
    # ====================================================================================
    print(f"\n{'=' * 60}\nFINAL COMBINATORIAL ANALYSIS (All Modes)\n{'=' * 60}")

    ensemble_results = []
    MODES = ['union', 'intersection', 'majority']

    # Only iterate over M values that didn't crash
    for sg_m in sg_data.keys():
        for dino_m in dino_data.keys():
            total_inf_time = sg_data[sg_m]['time'] + dino_data[dino_m]['time'] + clip_time
            for k in TOP_K_SEARCH:
                # Slice pre-computed ranks
                SG_top = retrieve_top_k(cfg, sg_data[sg_m]['ranks'], k, 'SuperGlobal', True)
                DINO_top = retrieve_top_k(cfg, dino_data[dino_m]['ranks'], k, 'DINOv2', True)
                CLIP_top = retrieve_top_k(cfg, clip_ranks, k, 'CLIP', True)

                models = [['SuperGlobal', SG_top], ['DINOv2', DINO_top], ['CLIP', CLIP_top]]

                for mode in MODES:
                    merged_res = merge_results(cfg, models, mode)
                    m = evaluate_final(cfg, models, merged_res, mode, silent=True)

                    ensemble_results.append({
                        'mode': mode,
                        'sg_m': sg_m,
                        'sg_map': sg_data[sg_m]['mAP'],  # Individual SG Accuracy
                        'dino_m': dino_m,
                        'dino_map': dino_data[dino_m]['mAP'],  # Individual DINO Accuracy
                        'top_k': k,
                        'precision': m['precision'],
                        'recall': m['recall'],
                        'f3': m['f3'],
                        'total_time': total_inf_time
                    })

    # --- FIND THE BEST PATH TO 20/50 ---
    print(f"\n{'*' * 40}\nCONFIGURATIONS MEETING TARGET (P>=0.20, R>=0.50)\n{'*' * 40}")

    targets_met = [r for r in ensemble_results if r['precision'] >= 0.20 and r['recall'] >= 0.50]

    if targets_met:
        # Sort by Recall descending to see your best possible capture first
        sorted_targets = sorted(targets_met, key=lambda x: x['recall'], reverse=True)

        for res in sorted_targets[:20]:  # Print top 20 best recall that met precision
            print(f"[{res['mode'].upper()}] SG_M: {res['sg_m']}, DINO_M: {res['dino_m']}, K: {res['top_k']} | "
                  f"P: {res['precision']:.2%}, R: {res['recall']:.2%}, F3: {res['f3']:.4f} | "
                  f"SG_mAP: {res['sg_map']:.2f}, DINO_mAP: {res['dino_map']:.2f} | "
                  f"Time: {res['total_time']:.2f}s")
    else:
        print("No configuration met the 20/50 target on this dataset.")

    # --- 5. EXPORT TO CSV FOR THESIS PLOTTING ---
    csv_filename = f"grid_search_{c.TEST.DATASET}_{int(time.time())}.csv"

    print(f"\n>> Exporting all {len(ensemble_results)} combinations to {csv_filename}...")

    keys = ensemble_results[0].keys() if ensemble_results else []
    if keys:
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(ensemble_results)
        print(">> Export complete.")


if __name__ == "__main__":
    main()