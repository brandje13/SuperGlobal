import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import torch
import numpy as np

from model.SuperGlobal.utils.SG_utils import extract_feature, test_revisitop
from model.SuperGlobal.modules.reranking.MDescAug import MDescAug
from model.SuperGlobal.modules.reranking.RerankwMDA import RerankwMDA


@torch.no_grad()
def test_model(model, device, cfg, gnd, data_dir, dataset, scale_list, custom, update_data, update_queries,
               top_m_rerank, is_rerank, gemp, rgem, sgem, onemeval, depth, evaluate, logger, model_id):
    torch.backends.cudnn.benchmark = False
    model.eval()
    torch.cuda.set_device(device)
    state_dict = model.state_dict()

    MDescAug_obj = MDescAug(M=top_m_rerank, K=9)
    RerankwMDA_obj = RerankwMDA(M=top_m_rerank, K=9)

    model.load_state_dict(state_dict)

    safe_model_name = str(model_id).split('\\')[-1].split('/')[-1].replace('.pyth', '').replace('.pth', '')
    text = f'>> {dataset}: Global Retrieval for scale {scale_list} with CVNet-Global ({safe_model_name})'
    print(text)

    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, f"SG_query_{safe_model_name}.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_feature(model, data_dir, dataset, gnd, "query", [1.0], gemp, rgem, sgem, scale_list)
        torch.save(Q, Q_path)
    else:
        Q = torch.load(Q_path)

    print("extract database features")
    X_path = os.path.join(data_dir, dataset, f"SG_data_{safe_model_name}.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_feature(model, data_dir, dataset, gnd, "db", [1.0], gemp, rgem, sgem, scale_list)
        torch.save(X, X_path)
    else:
        X = torch.load(X_path)

    Q_tensor = torch.from_numpy(Q).float().to(device)
    X_tensor = torch.from_numpy(X).float().to(device)

    print("perform global feature reranking")
    if onemeval:
        X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
        X = torch.cat([X, X_expand], 0)

    index = faiss.IndexFlatIP(2048)
    index.add(X_tensor.cpu().numpy())

    dist, inx = index.search(Q_tensor.cpu().numpy(), len(X_tensor))
    ranks = inx.T

    if is_rerank and top_m_rerank > 0:
        ranks = torch.from_numpy(ranks).to(device)
        rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X_tensor, Q_tensor, ranks)
        ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy()

    mapE = 0.0

    if evaluate:
        ks = [10, 25, 100]
        if not custom:
            (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
            print('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                         np.around(mapM * 100, decimals=2),
                                                                         np.around(mapH * 100, decimals=2)))

    return ranks, mapE