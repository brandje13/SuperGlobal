# written by Seongwon Lee (won4113@yonsei.ac.kr)
import os

import config as config
import model.SuperGlobal.CVNet_tester as CVNet_tester
from tkfilebrowser import askopenfilenames, askopendirname

from config import cfg as c
from model.ConvNeXtV2 import ConvNeXtV2_tester
from model.MixVPR import MixVPR_tester
from model.SAM import SAM_tester
from model.DINOv2 import DINO_tester
from model.CLIP import CLIP_tester
from model.SigLIP import SigLIP_tester
from utils.config_gnd import config_gnd
from utils.evaluate_final import evaluate_final
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_top_k, save_merged_results
from utils.merge_results import merge_results


def main():
    config.load_cfg_fom_args("utils a CVNet model.")
    c.NUM_GPUS = 1
    c.freeze()

    if c.TEST.CUSTOM:
        query_paths = askopenfilenames()
        data_dir = askopendirname()
        create_groundtruth(query_paths, data_dir, c.TEST.DATASET)  # TODO: Fix custom dataset param
        gnd = 'custom.json'
        dataset = "custom"
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

    SG_ranks, SG_map = CVNet_tester.__main__(gnd, cfg)
    SG_top = retrieve_top_k(cfg, SG_ranks, c.TEST.TOP_K, 'SuperGlobal', False)

    DINO_ranks, DINO_map = DINO_tester.__main__(gnd, cfg)
    DINO_top = retrieve_top_k(cfg, DINO_ranks, c.TEST.TOP_K, 'DINOv2', False)

    SigLIP_ranks, SigLIP_map = SigLIP_tester.__main__(gnd, cfg)
    SigLIP_top = retrieve_top_k(cfg, SigLIP_ranks, c.TEST.TOP_K, 'SigLIP', False)

    models = [['SuperGlobal', SG_top], ['DINOv2', DINO_top], ['SigLIP', SigLIP_top]]

    results_union = merge_results(cfg, models, 'union')
    results_intersection = merge_results(cfg, models, 'intersection')
    results_majority = merge_results(cfg, models, 'majority')

    save_merged_results(cfg, results_union, models, 'union')
    save_merged_results(cfg, results_intersection, models, 'intersection')
    save_merged_results(cfg, results_majority, models, 'majority')

    evaluate_final(cfg, models, results_union, 'union')
    evaluate_final(cfg, models, results_intersection, 'intersection')
    evaluate_final(cfg, models, results_majority, 'majority')


if __name__ == "__main__":
    main()
