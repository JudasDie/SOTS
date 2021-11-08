''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: read files with [.yaml] [.txt]
Data: 2021.6.23
In case ray error, pls use: export PYTHONPATH=${PYTHONPATH}:$YourPath/SOTS:$YourPath/SOTS/lib
'''

import _init_paths
import os
import argparse
import numpy as np
from pprint import pprint
from easydict import EasyDict as edict

import ray
from ray import tune
from hyperopt import hp
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import utils.read_file as reader
import utils.model_helper as loader
import utils.sot_builder as builder
import utils.tune_helper as tuner_build
import tracker.sot_tracker as tracker_builder

parser = argparse.ArgumentParser(description='parameters for Ocean tracker')
parser.add_argument('--cfg', type=str, default='experiments/SiamDW.yaml', help='yaml configure file name')
parser.add_argument('--cache_dir', default='./TPE_results', type=str, help='directory to store cache')
parser.add_argument('--resume', default=None, type=str, help='resume checkpoint')

args = parser.parse_args()

# parser config
print('====> parser configs <====')
print('[*] the config file is: {}'.format(args.cfg))
config = edict(reader.load_yaml(args.cfg))

resume = config.TUNE.RESUME
if resume is None or resume == 'None': raise ValueError('please set resume checkpoint in config file {}'.format(resume))

curdir = os.path.abspath(os.path.dirname(__file__))
resume = os.path.join(curdir, '..', resume)

# print build tuner
print('====> build tuner <====')
tuner = tuner_build.SOTtuner_builder(config)


# fitness function
def fitness(params, reporter):
    if 'Siam' in config.MODEL.NAME or config.MODEL.NAME in ['Ocean', 'OceanPlus', 'AutoMatch', 'TransT']:
        tracker = tracker_builder.SiamTracker(config)
        siambuilder = builder.Siamese_builder(config)
        model = siambuilder.build()
    else:
        raise Exception('Not implemented model type!')

    # print(model)
    model = loader.load_pretrain(model, resume)
    model.eval()
    model = model.cuda()

    print('pretrained model has been loaded')
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.MODEL.NAME in ['Ocean', 'AutoMatch']:
        penalty_k = params["penalty_k"]
        lr = params["lr"]  # scale
        window_influence = params["window_influence"]
        small_sz = params["small_sz"]
        big_sz = params["big_sz"]

        model_config = dict()
        model_config['config'] = config
        model_config['hp'] = dict()
        model_config['hp']['penalty_k'] = penalty_k
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = lr
        model_config['hp']['small_sz'] = small_sz
        model_config['hp']['big_sz'] = big_sz
    elif config.MODEL.NAME in ['SiamFC', 'SiamDW']:
        scale_step_FC = params["scale_step_FC"]
        scale_penalty_FC = params["scale_penalty_FC"]
        lr = params["lr"]
        window_influence = params["window_influence"]
        model_config = dict()
        model_config['config'] = config
        model_config['hp'] = dict()
        model_config['hp']['scale_step_FC'] = scale_step_FC
        model_config['hp']['scale_penalty_FC'] = scale_penalty_FC
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = lr
    else:
        raise ValueError('not supported tracker')

    # tuning
    inputs = {'model': model, 'tracker': tracker, 'config': model_config}
    score = tuner.run(inputs)
    reporter(score=score)
    # tune.report(score=score)


if __name__ == "__main__":
    gpu_nums = len([int(i) for i in config.COMMON.GPUS.split(',')])
    ray.init(num_gpus=gpu_nums, num_cpus=gpu_nums * 8,  object_store_memory=500000000)
    # ray.init(num_gpus=gpu_nums, num_cpus=gpu_nums * 8)
    tune.register_trainable("fitness", fitness)

    if config.MODEL.NAME in ['Ocean', 'AutoMatch']:
        params = {
                "penalty_k": hp.quniform('penalty_k', 0.001, 0.2, 0.001),
                "lr": hp.quniform('lr', 0.3, 0.8, 0.001),
                "window_influence": hp.quniform('window_influence', 0.15, 0.65, 0.001),
                "small_sz": hp.choice("small_sz", [255, 271]),
                "big_sz": hp.choice("big_sz", [271, 287]),
                }
    elif config.MODEL.NAME in ['OceanPlus']:
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
    elif config.MODEL.NAME in ['SiamFC', 'SiamDW']:
        params = {
            "scale_step_FC": hp.quniform('scale_step_FC', 1.0, 1.2, 0.001),
            "scale_penalty_FC": hp.quniform('scale_penalty_FC', 0.95, 1.0, 0.001),
            "window_influence": hp.quniform('window_influence', 0.2, 0.7, 0.001),
            "lr": hp.quniform("lr", 0.2, 0.9, 0.001),
        }
    else:
        raise ValueError('not supported tracker')

    print('tuning range: ')
    pprint(params)

    tune_spec = {
        "tpe_tune": {
            "run": "fitness",
            "resources_per_trial": {
                "cpu": 1,  # single task cpu num
                "gpu": 1.0 / 1.0 / config.TUNE.TRIAL_PER_GPU,  # single task gpu num
            },
            "num_samples": 10000,  # sample hyperparameters times
            "local_dir": args.cache_dir
        }
    }



    scheduler = AsyncHyperBandScheduler(
        # time_attr="timesteps_total",
        metric='score',
        mode='max',
        max_t=400,
        grace_period=20
    )

    # stop condition for VOT and OTB
    if config.TUNE.DATA.startswith('VOT'):
        stop = {
            "score": 0.6,  # if EAO >= 0.6, this procedures will stop
            # "timesteps_total": 100, # iteration times
        }

        tune_spec['tpe_tune']['stop'] = stop

        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            metric='score',
            mode='max',
            max_t=400,
            grace_period=20
        )

        algo = HyperOptSearch(params, max_concurrent=gpu_nums * config.TUNE.TRIAL_PER_GPU + 1, metric='score', mode='max')

    elif config.TUNE.DATA in ['OTB2015', 'GOT10KVAL',  'GOT10KTEST', 'TOTB', 'NFS30', 'TC128', 'UAV123']:
        stop = {
            # "timesteps_total": 100, # iteration times
            "score": 0.90  # AUC
        }

        tune_spec['tpe_tune']['stop'] = stop
        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="score",
            max_t=400,
            grace_period=20
        )

        algo = HyperOptSearch(params, max_concurrent=gpu_nums * config.TUNE.TRIAL_PER_GPU + 1, metric='score', mode='max')

    else:
        raise ValueError("not support other dataset now")


    tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)




