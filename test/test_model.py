# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import torch
import numpy as np
from tkfilebrowser import askopenfilenames, askopendirname

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop, print_top_n, create_groundtruth, process_txt_files
from test.dataset import DataSet

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA

import fiftyone as fo


@torch.no_grad()
def test_model(model, device, data_dir, dataset_list, scale_list, custom, update_data, update_queries,
               is_rerank, gemp, rgem, sgem, onemeval, depth, logger):
    torch.backends.cudnn.benchmark = False
    model.eval()
    torch.cuda.set_device(device)
    
    state_dict = model.state_dict()

    # initialize modules
    MDescAug_obj = MDescAug(M=600, K=9)
    RerankwMDA_obj = RerankwMDA(M=600, K=9)

    model.load_state_dict(state_dict)

    for dataset in [dataset_list]:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if custom:
            query_paths = askopenfilenames()
            data_dir = askopendirname()
            create_groundtruth(query_paths, data_dir, dataset) # TODO: Fix custom dataset param
            gnd_fn = 'custom.pkl'
            dataset = "custom"
        elif dataset in ['roxford5k', 'rparis6k']:
            gnd_fn = f'gnd_{dataset}.pkl'
            file_path = os.path.join(data_dir, dataset)
            process_txt_files(data_dir, dataset)
        elif not dataset == "":
            query_paths = [os.path.join(data_dir, dataset, "queries", i) for i in os.listdir(os.path.join(data_dir, dataset, "queries"))]
            create_groundtruth(query_paths, data_dir, dataset)
            file_path = os.path.join(data_dir, dataset)
            gnd_fn = f'gnd_{dataset}.pkl'
        else:
            file_path = ''
            assert dataset


        cfg = config_gnd(dataset, data_dir, custom)
        print(cfg)

        print("extract query features")
        Q_path = os.path.join(data_dir, dataset, "query_features.pt")
        if update_queries or not os.path.isfile(Q_path):
            Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
            torch.save(Q, Q_path)
        else:
            Q = torch.load(Q_path)


        print("extract database features")
        X_path = os.path.join(data_dir, dataset, "data_features.pt")
        if update_data or not os.path.isfile(X_path):
            X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)
            torch.save(X, X_path)
        else:
            X = torch.load(X_path)



        Q = torch.tensor(Q).cuda()
        X = torch.tensor(X).cuda()

        print("perform global feature reranking")
        if onemeval:
            X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
            X = torch.cat([X, X_expand], 0)
        sim = torch.matmul(X, Q.T)  # 6322 70
        ranks = torch.argsort(-sim, dim=0)  # 6322 70

        if is_rerank:
            rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
            ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy()

        print_top_n(cfg, ranks, 10, file_path)

        # # revisited evaluation
        # ks = [1, 5, 10]
        # if not custom:
        #     (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
        #
        #     print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE * 100, decimals=2),
        #                                                               np.around(mapM * 100, decimals=2),
        #                                                               np.around(mapH * 100, decimals=2)))
        #     logger.info('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE * 100, decimals=2),
        #                                                                     np.around(mapM * 100, decimals=2),
        #                                                                     np.around(mapH * 100, decimals=2)))
        dataset = fo.Dataset.from_images_dir(os.path.join(data_dir, dataset))

        session = fo.launch_app(dataset, desktop=True)
        session.wait()

        # print('>> {}: Reranking results with CVNet-Rerank'.format(dataset))
        #
        # gnd = cfg['gnd']
        # query_dataset = DataSet(data_dir, dataset, gnd_fn, "query", [1.0])
        # db_dataset = DataSet(data_dir, dataset, gnd_fn, "db", [1.0])
        # sim_corr_dict = {}
        # for topk in [100]:
        #     print("current top-k value: ", topk)
        #     for i in tqdm(range(int(cfg['nq']))):
        #         im_q = query_dataset.__getitem__(i)[0]
        #         im_q = torch.from_numpy(im_q).cuda().unsqueeze(0)
        #         feat_q = model.extract_featuremap(im_q)
        #
        #         rerank_count = np.zeros(3, dtype=np.uint16)
        #         for j in range(int(cfg['n'])):
        #             if (rerank_count >= topk).sum() == 3:
        #                 break
        #
        #             rank_j = ranks[j][i]
        #
        #             if rank_j in gnd[i]['junk']:
        #                 continue
        #             elif rank_j in gnd[i]['good']:
        #                 append_j = np.asarray([True, True, False])
        #             elif rank_j in gnd[i]['ok']:
        #                 append_j = np.asarray([False, True, True])
        #             else:  # negative
        #                 append_j = np.asarray([True, True, True])
        #
        #             append_j *= (rerank_count < topk)
        #
        #             if append_j.sum() > 0:
        #                 im_k = db_dataset.__getitem__(rank_j)[0]
        #                 im_k = torch.from_numpy(im_k).cuda().unsqueeze(0)
        #                 feat_k = model.extract_featuremap(im_k)
        #
        #                 score = model.extract_score_with_featuremap(feat_q, feat_k).cpu()
        #                 sim_corr_dict[(rank_j, i)] = score
        #                 rerank_count += append_j
        #
        #     mix_ratio = 0.5
        #     ranks_corr_list = rerank_ranks_revisitop(cfg, topk, ranks, sim, sim_corr_dict, mix_ratio)
        #     (mapE_r, apsE_r, mprE_r, prsE_r), (mapM_r, apsM_r, mprM_r, prsM_r), (
        #     mapH_r, apsH_r, mprH_r, prsH_r) = test_revisitop(cfg, ks, ranks_corr_list)
        #     print('Reranking results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE_r * 100, decimals=2),
        #                                                               np.around(mapM_r * 100, decimals=2),
        #                                                               np.around(mapH_r * 100, decimals=2)))
