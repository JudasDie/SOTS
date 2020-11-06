from __future__ import absolute_import
import _init_paths
import os
import argparse
import numpy as np

import models.models as models
from utils.utils import load_pretrain
from test_ocean import auc_otb, eao_vot
from test_oceanplus import eao_vot_oceanplus, auc_got10k_oceanplus
from test_siamdw import auc_otb_fc, eao_vot_fc

from tracker.ocean import Ocean
from tracker.oceanplus import OceanPlus
from tracker.siamfc import SiamFC
from easydict import EasyDict as edict

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
#from ray.tune.suggest import HyperOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from pprint import pprint

parser = argparse.ArgumentParser(description='parameters for Ocean tracker')
parser.add_argument('--arch', dest='arch', default='Ocean',
                    help='architecture of model')
parser.add_argument('--resume', default='', type=str, required=True,
                    help='resumed model')
parser.add_argument('--cache_dir', default='./TPE_results', type=str, help='directory to store cache')
parser.add_argument('--gpu_nums', default=8, type=int, help='gpu numbers')
parser.add_argument('--trial_per_gpu', default=8, type=int, help='trail per gpu')
parser.add_argument('--dataset', default='OTB2013', type=str, help='dataset')
parser.add_argument('--align', default='True', type=str, help='align')
parser.add_argument('--online', default=False, type=bool, help='online flag')
parser.add_argument('--MMS', default=False, type=bool, help='MMS flag')

args = parser.parse_args()

# prepare tracker 
info = edict()
info.arch = args.arch
info.dataset = args.dataset
info.epoch_test = False

if args.arch in ['Ocean']:
    if args.online:
        info.align = False
    else:
        info.align = True if 'VOT' in args.dataset and args.align=='True' else False
    info.online = args.online
    info.TRT = 'TRT' in args.arch
    if info.TRT:
        info.align = False
elif args.arch in ['OceanPlus']:
    info.align, info.TRT, info.online = False, False, False
elif args.arch in ['SiamDW']:
    pass
else:
    raise ValueError('Unknown Tracker for TPE Tune')

args.resume = os.path.abspath(args.resume)


# fitness function
def fitness(config, reporter):
    # create model
    if args.arch in ['Ocean']:
        model = models.__dict__[args.arch](align=info.align)
        tracker = Ocean(info)
    elif args.arch in ['OceanPlus']:
        model = models.__dict__[args.arch](online=args.online, mms=args.MMS)
        tracker = OceanPlus(info)
    elif 'SiamDW' in args.arch:
        model = models.__dict__[args.arch]()
        tracker = SiamFC(info)
    else:
        raise ValueError('not supported other model now')

    model = load_pretrain(model, args.resume)
    model.eval()
    model = model.cuda()
    print('pretrained model has been loaded')
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    if args.arch in ['Ocean']:
        penalty_k = config["penalty_k"]
        scale_lr = config["scale_lr"]
        window_influence = config["window_influence"]
        small_sz = config["small_sz"]
        big_sz = config["big_sz"]
        ratio = config["ratio"]

        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['penalty_k'] = penalty_k
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = scale_lr
        model_config['hp']['small_sz'] = small_sz
        model_config['hp']['big_sz'] = big_sz
        model_config['hp']['ratio'] = ratio
    elif args.arch in ['OceanPlus']:
        penalty_k = config["penalty_k"]
        scale_lr = config["scale_lr"]
        window_influence = config["window_influence"]
        small_sz = config["small_sz"]
        big_sz = config["big_sz"]
        lambda_u = config["lambda_u"]
        lambda_s = config["lambda_s"]
        cyclic_thr = config["cyclic_thr"]
        choose_thr = config["choose_thr"]

        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['penalty_k'] = penalty_k
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = scale_lr
        model_config['hp']['small_sz'] = small_sz
        model_config['hp']['big_sz'] = big_sz
        model_config['hp']['lambda_u'] = lambda_u
        model_config['hp']['lambda_s'] = lambda_s
        model_config['hp']['cyclic_thr'] = cyclic_thr
        model_config['hp']['choose_thr'] = choose_thr

    elif args.arch in ['SiamFC', 'SiamDW']:
        scale_step = config["scale_step"]
        scale_penalty = config["scale_penalty"]
        scale_lr = config["scale_lr"]
        w_influence = config["w_influence"]
        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['scale_step'] = scale_step
        model_config['hp']['scale_penalty'] = scale_penalty
        model_config['hp']['w_influence'] = w_influence
        model_config['hp']['scale_lr'] = scale_lr
    else:
        raise ValueError('not supported tracker')

    # -------------
    # For OceanPlus
    # -------------
    # VOT and OceanPlus
    if args.dataset.startswith('VOT') and args.arch in ['OceanPlus']:
        eao = eao_vot_oceanplus(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, lambda_u: {5}, lambda_s: {6}, cyclic_thr: {7}, choose_thr: {8}, eao: {9}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, lambda_u, lambda_s, cyclic_thr, choose_thr, eao))
        reporter(EAO=eao)

    # GOT10K and OceanPlus
    if args.dataset.startswith('GOT') and args.arch in ['OceanPlus']:
        auc = auc_got10k_oceanplus(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, lambda_u: {5}, lambda_s: {6}, cyclic_thr: {7}, choose_thr: {8}, eao: {9}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, lambda_u, lambda_s, cyclic_thr, choose_thr, auc.item()))
        reporter(AUC=auc)

    # ----------
    # For Ocean
    # ----------
    # VOT and Ocean
    if args.dataset.startswith('VOT') and args.arch in ['Ocean']:
        eao = eao_vot(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, ratio: {6}, eao: {5}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, eao, ratio))
        reporter(EAO=eao)

    # OTB and Ocean
    if args.dataset.startswith('OTB') and args.arch in ['Ocean']:
        auc = auc_otb(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, ratio: {6}, eao: {5}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, auc.item(), ratio))
        reporter(AUC=auc)

    # ------------
    # For SiamDW
    # ------------
    # OTB and SiamFC/DW
    if args.dataset.startswith('OTB') and ('SiamFC' in args.arch or 'SiamDW' in args.arch):
        auc = auc_otb_fc(tracker, model, model_config)
        print("scale_step: {0}, scale_lr: {1}, scale_penalty: {2}, window_influence: {3}, auc: {4}".format(scale_step, scale_lr, scale_penalty, w_influence, auc.item()))
        reporter(AUC=auc)

    # VOT and SiamFC/DW
    if args.dataset.startswith('VOT') and ('SiamFC' in args.arch or 'SiamDW' in args.arch):
        eao = eao_vot_fc(tracker, model, model_config)
        print("scale_step: {0}, scale_lr: {1}, scale_penalty: {2}, window_influence: {3}, eao: {4}".format(scale_step, scale_lr, scale_penalty, w_influence, eao))
        reporter(EAO=eao)


if __name__ == "__main__":
    # the resources you computer have, object_store_memory is shm
    #ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8,  object_store_memory=50000000000)
    ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8,  object_store_memory=500000000)
    tune.register_trainable("fitness", fitness)

    if args.arch in ['Ocean']:
        params = {
                "penalty_k": hp.quniform('penalty_k', 0.001, 0.2, 0.001),
                "scale_lr": hp.quniform('scale_lr', 0.3, 0.8, 0.001),
                "window_influence": hp.quniform('window_influence', 0.15, 0.65, 0.001),
                "small_sz": hp.choice("small_sz", [255, 271]),
                "big_sz": hp.choice("big_sz", [271, 287]),
                "ratio": hp.quniform('ratio', 0.7, 1, 0.01),
                }
    elif args.arch in ['OceanPlus']:
        params = {
            # "penalty_k": hp.choice('penalty_k', [0.06]),
            # "scale_lr": hp.choice('scale_lr', [0.644]),
            # "window_influence": hp.choice('window_influence', [0.484]),
            # "small_sz": hp.choice("small_sz", [255]),
            # "big_sz": hp.choice("big_sz", [287]),
            "penalty_k": hp.choice('penalty_k', [0.022]),
            "scale_lr": hp.choice('scale_lr', [0.799]),
            "window_influence": hp.choice('window_influence', [0.118]),
            "small_sz": hp.choice("small_sz", [255]),
            "big_sz": hp.choice("big_sz", [255]),
            "lambda_u": hp.quniform('lambda_u', 0.1, 1, 0.05),
            "lambda_s": hp.quniform('lambda_s', 0.1, 1, 0.05),
            "cyclic_thr": hp.quniform('cyclic_thr', 0.7, 1, 0.01),
            "choose_thr": hp.quniform('choose_thr', 0.4, 0.8, 0.01),
        }
    elif 'SiamFC' in args.arch or 'SiamDW' in args.arch:
        params = {
            "scale_step": hp.quniform('scale_step', 1.0, 1.2, 0.0001),
            "scale_penalty": hp.quniform('scale_penalty', 0.95, 1.0, 0.0001),
            "w_influence": hp.quniform('w_influence', 0.2, 0.7, 0.0001),
            "scale_lr": hp.quniform("scale_lr", 0.2, 0.9, 0.0001),
        }
    else:
        raise ValueError('not supported tracker')

    if ('VOT' not in args.dataset or not args.align) and args.arch in ['Ocean']:
        params['ratio'] = hp.choice("ratio", [1]) 
 
    print('tuning range: ')
    pprint(params)    

    tune_spec = {
        "zp_tune": {
            "run": "fitness",
            "resources_per_trial": {
                "cpu": 1,  # single task cpu num
                "gpu": 1.0 / args.trial_per_gpu,  # single task gpu num
            },
            "num_samples": 10000,  # sample hyperparameters times
            "local_dir": args.cache_dir
        }
    }

    # stop condition for VOT and OTB
    if args.dataset.startswith('VOT'):
        stop = {
            "EAO": 0.6,  # if EAO >= 0.6, this procedures will stop
            # "timesteps_total": 100, # iteration times
        }
        tune_spec['zp_tune']['stop'] = stop

        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            metric='EAO',
            mode='max',
            max_t=400,
            grace_period=20
        )
        # max_concurrent: the max running task
        #algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*args.trial_per_gpu + 1, reward_attr="EAO")
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*args.trial_per_gpu + 1, metric='EAO',  mode='max')

    elif args.dataset.startswith('OTB') or args.dataset.startswith('VIS') or args.dataset.startswith('GOT10K'):
        stop = {
            # "timesteps_total": 100, # iteration times
            "AUC": 0.80
        }
        tune_spec['zp_tune']['stop'] = stop
        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="AUC",
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums * args.trial_per_gpu + 1, metric='AUC', mode='max')
        # algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*2 + 1, reward_attr="AUC")  #
    else:
        raise ValueError("not support other dataset now")

    tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)



