import os
import time
import torch
import csv
import gc
from tqdm import tqdm

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

    # --- 2. DEFINE FULL ARCHITECTURE SEARCH SPACE ---
    SG_BACKBONES = ['.\\weights\\CVPR2022_CVNet_R50.pyth', '.\\weights\\CVPR2022_CVNet_R101.pyth']
    DINO_BACKBONES = [
        'vit_small_patch14_dinov2.lvd142m',
        'vit_base_patch14_dinov2.lvd142m',
        'vit_large_patch14_dinov2.lvd142m'
    ]
    CLIP_BACKBONES = ['openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14']

    SG_M_SEARCH = list(range(1000, 11000, 1000))
    DINO_M_SEARCH = list(range(1000, 2000, 1000))
    TOP_K_SEARCH = list(range(10, 160, 10))

    sg_data = {}
    dino_data = {}
    clip_data = {}

    # ====================================================================================
    # PHASE 1: SuperGlobal
    # ====================================================================================
    for sg_bb in SG_BACKBONES:
        print(f"\n{'=' * 40}\n>> Initializing SuperGlobal with {sg_bb}\n{'=' * 40}")
        c.SupG.WEIGHTS = sg_bb

        for m in SG_M_SEARCH:
            c.SupG.TOP_M = m
            start = time.time()

            try:
                ranks, mAP = CVNet_tester.__main__(gnd, cfg)
                elapsed = time.time() - start
                sg_data[(sg_bb, m)] = {'ranks': ranks, 'mAP': mAP, 'time': elapsed}

            except torch.cuda.OutOfMemoryError:
                print(f"\n[!] WARNING: CUDA OOM at SuperGlobal {sg_bb} M={m}. Skipping.")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[!] WARNING: CUDA OOM at SuperGlobal {sg_bb} M={m}. Skipping.")
                else:
                    raise e

            finally:
                torch.cuda.empty_cache()
                gc.collect()

        torch.cuda.empty_cache()
        gc.collect()

    # ====================================================================================
    # PHASE 2: DINOv2
    # ====================================================================================
    for dino_bb in DINO_BACKBONES:
        print(f"\n{'=' * 40}\n>> Initializing DINOv2 with {dino_bb}\n{'=' * 40}")
        c.DINO.WEIGHTS = dino_bb

        for m in DINO_M_SEARCH:
            c.DINO.TOP_M = m
            start = time.time()

            try:
                ranks, mAP = DINO_tester.__main__(gnd, cfg)
                elapsed = time.time() - start
                dino_data[(dino_bb, m)] = {'ranks': ranks, 'mAP': mAP, 'time': elapsed}

            except torch.cuda.OutOfMemoryError:
                print(f"\n[!] WARNING: CUDA OOM at DINOv2 {dino_bb} M={m}. Skipping.")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[!] WARNING: CUDA OOM at DINOv2 {dino_bb} M={m}. Skipping.")
                else:
                    raise e

            finally:
                torch.cuda.empty_cache()
                gc.collect()

        torch.cuda.empty_cache()
        gc.collect()

    # ====================================================================================
    # PHASE 3: CLIP
    # ====================================================================================
    for clip_bb in CLIP_BACKBONES:
        print(f"\n{'=' * 40}\n>> Initializing CLIP with {clip_bb}\n{'=' * 40}")
        c.CLIP.WEIGHTS = clip_bb
        start = time.time()

        try:
            ranks, mAP = CLIP_tester.__main__(gnd, cfg)
            elapsed = time.time() - start
            clip_data[clip_bb] = {'ranks': ranks, 'mAP': mAP, 'time': elapsed}
        except Exception as e:
            print(f"\n[!] Error running CLIP {clip_bb}: {e}. Skipping.")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        # ====================================================================================
        # PHASE 4: THE FULL COMBINATORIAL FRONTIER
        # ====================================================================================
        MODES = ['union', 'intersection', 'majority']
        total_combos = len(sg_data) * len(dino_data) * len(clip_data) * len(TOP_K_SEARCH) * len(MODES)
        print(f"\n{'=' * 60}\nFINAL COMBINATORIAL ANALYSIS ({total_combos} combinations)\n{'=' * 60}")
        ensemble_results = []

        with tqdm(total=total_combos, desc="Fusing Ensembles", unit="combo") as pbar:
            for (sg_bb, sg_m), sg_info in sg_data.items():
                for (dino_bb, dino_m), dino_info in dino_data.items():
                    for clip_bb, clip_info in clip_data.items():

                        total_inf_time = sg_info['time'] + dino_info['time'] + clip_info['time']

                        for k in TOP_K_SEARCH:
                            SG_top = retrieve_top_k(cfg, sg_info['ranks'], k, 'SuperGlobal', False)
                            DINO_top = retrieve_top_k(cfg, dino_info['ranks'], k, 'DINOv2', False)
                            CLIP_top = retrieve_top_k(cfg, clip_info['ranks'], k, 'CLIP', False)

                            models = [['SuperGlobal', SG_top], ['DINOv2', DINO_top], ['CLIP', CLIP_top]]

                            for mode in MODES:
                                merged_res = merge_results(cfg, models, mode)
                                m_metrics = evaluate_final(cfg, models, merged_res, mode, silent=True)

                                ensemble_results.append({
                                    'mode': mode,
                                    'sg_bb': sg_bb,
                                    'dino_bb': dino_bb,
                                    'clip_bb': clip_bb,
                                    'sg_m': sg_m,
                                    'sg_map': sg_info['mAP'],
                                    'dino_m': dino_m,
                                    'dino_map': dino_info['mAP'],
                                    'clip_map': clip_info['mAP'],
                                    'top_k': k,
                                    'precision': m_metrics['precision'],
                                    'recall': m_metrics['recall'],
                                    'f3': m_metrics['f3'],
                                    'total_time': total_inf_time
                                })

                                # Tick the progress bar forward by 1
                                pbar.update(1)

    # --- FIND THE BEST PATH TO 20/50 ---
    print(f"\n{'*' * 40}\nCONFIGURATIONS MEETING TARGET (P>=0.20, R>=0.50)\n{'*' * 40}")

    targets_met = [r for r in ensemble_results if r['precision'] >= 0.20 and r['recall'] >= 0.50]

    if targets_met:
        # Sort by F3 Score
        sorted_targets = sorted(targets_met, key=lambda x: x['f3'], reverse=True)

        for res in sorted_targets[:20]:
            # String slicing to make the printout readable in the console
            sg_short = res['sg_bb'].split('_')[-1].split('.')[0]
            dino_short = res['dino_bb'].split('_')[1]
            clip_short = res['clip_bb'].split('-')[-2]

            print(
                f"[{res['mode'].upper()}] K:{res['top_k']} | SG:{sg_short}({res['sg_m']}), DINO:{dino_short}({res['dino_m']}), CLIP:{clip_short} | "
                f"P:{res['precision']:.2%}, R:{res['recall']:.2%}, F3:{res['f3']:.4f} | "
                f"mAPs [SG:{res['sg_map']:.2f}, DI:{res['dino_map']:.2f}, CL:{res['clip_map']:.2f}] | "
                f"Time:{res['total_time']:.1f}s")
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
        print(f">> Export complete! Data saved to {os.getcwd()}\\{csv_filename}")


if __name__ == "__main__":
    main()