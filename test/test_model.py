# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import torch
import numpy as np
from tkfilebrowser import askopenfilenames, askopendirname
import time

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop, print_top_n, create_groundtruth

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA
import torch 
@torch.no_grad()
def test_model(model, device, data_dir, dataset_list, scale_list, custom, is_rerank, gemp, rgem, sgem, onemeval, depth, logger):
    torch.backends.cudnn.benchmark = False
    model.eval()
    torch.cuda.set_device(device)
    
    state_dict = model.state_dict()
    

    # initialize modules
    MDescAug_obj = MDescAug(M=600, K=9)
    RerankwMDA_obj = RerankwMDA(M=600, K=9)



    model.load_state_dict(state_dict)
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if custom:
            query_paths = askopenfilenames()
            data_dir = askopendirname()
            create_groundtruth(query_paths, data_dir)
            gnd_fn = 'custom.pkl'
            dataset = "custom"
        elif dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
            file_path = 'revisitop/roxford5k/jpg/'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        elif dataset == 'smartTrim':
            query_paths = [i for i in os.listdir(data_dir + 'queries/')]
            #dir_path = '/home/nick/Downloads/data/'
            create_groundtruth(query_paths, data_dir)
            gnd_fn = 'gnd_smartTrim.pkl'
        else:
            file_path = ''
            assert dataset
        print("extract query features")
        Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
        print("extract database features")
        X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)

        cfg = config_gnd(dataset,data_dir,custom)
        Q = torch.tensor(Q).cuda()
        X = torch.tensor(X).cuda()

        print("perform global feature reranking")
        if onemeval:
            X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
            X = torch.cat([X,X_expand],0)
        sim = torch.matmul(X, Q.T) # 6322 70
        ranks = torch.argsort(-sim, axis=0) # 6322 70

        if is_rerank:
            rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
            ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy()

        print_top_n(cfg, ranks, 20, file_path)

        # revisited evaluation
        ks = [1, 5, 10]
        if not custom:
            (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

            print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
            logger.info('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        
