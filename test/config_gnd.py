# Original code: revisitop (https://github.com/filipradenovic/revisitop)

import os
import json

DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m', "smartTrim", "catndogs"]

def config_gnd(dataset, dir_main, custom):

    #dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if not dataset == 'revisitop1m':
        # loading imlist, qimlist, and gnd, in cfg as a dict
        if custom:
            gnd_fname = os.path.join(dir_main, dataset, 'custom.json')
        else:
            gnd_fname = os.path.join(dir_main, dataset, f'gnd_{dataset}.json')

        with open(gnd_fname, 'rb') as f:
            cfg = json.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

    elif dataset == 'revisitop1m':
        # loading imlist from a .txt file
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, f'{dataset}.txt')
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'])

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i])

def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist
