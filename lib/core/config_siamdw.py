# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email:  houwen.peng@microsoft.com
# Details: This script provides configs for siamfc and siamrpn
# ------------------------------------------------------------------------------

import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0,1,2,3"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

# #-----————- config for siamfc ------------
config.SIAMDW = edict()
config.SIAMDW.TRAIN = edict()
config.SIAMDW.TEST = edict()
config.SIAMDW.TUNE = edict()
config.SIAMDW.DATASET = edict()
config.SIAMDW.DATASET.VID = edict()       # paper utlized but not recommended
config.SIAMDW.DATASET.GOT10K = edict()    # not utlized in paper but recommended, better performance and more stable

# augmentation
config.SIAMDW.DATASET.SHIFT = 4
config.SIAMDW.DATASET.SCALE = 0.05
config.SIAMDW.DATASET.COLOR = 1
config.SIAMDW.DATASET.FLIP = 0
config.SIAMDW.DATASET.BLUR = 0
config.SIAMDW.DATASET.ROTATION = 0

# vid
config.SIAMDW.DATASET.VID.PATH = '/home/zhbli/Dataset/data2/vid/crop511'
config.SIAMDW.DATASET.VID.ANNOTATION = '/home/zhbli/Dataset/data2/vid/train.json'

# got10k
config.SIAMDW.DATASET.GOT10K.PATH = '/home/zhbli/Dataset/data3/got10k/crop511'
config.SIAMDW.DATASET.GOT10K.ANNOTATION = '/home/zhbli/Dataset/data3/got10k/train.json'

# train
config.SIAMDW.TRAIN.MODEL = "SiamFCIncep22"
config.SIAMDW.TRAIN.RESUME = False
config.SIAMDW.TRAIN.START_EPOCH = 0
config.SIAMDW.TRAIN.END_EPOCH = 50
config.SIAMDW.TRAIN.TEMPLATE_SIZE = 127
config.SIAMDW.TRAIN.SEARCH_SIZE = 255
config.SIAMDW.TRAIN.STRIDE = 8
config.SIAMDW.TRAIN.BATCH = 32
config.SIAMDW.TRAIN.PAIRS = 200000
config.SIAMDW.TRAIN.PRETRAIN = 'resnet23_inlayer.model'
config.SIAMDW.TRAIN.LR_POLICY = 'log'
config.SIAMDW.TRAIN.LR = 0.001
config.SIAMDW.TRAIN.LR_END = 0.0000001
config.SIAMDW.TRAIN.MOMENTUM = 0.9
config.SIAMDW.TRAIN.WEIGHT_DECAY = 0.0001
config.SIAMDW.TRAIN.WHICH_USE = 'GOT10K'  # VID or 'GOT10K'

# test
config.SIAMDW.TEST.MODEL = config.SIAMDW.TRAIN.MODEL
config.SIAMDW.TEST.DATA = 'VOT2015'
config.SIAMDW.TEST.START_EPOCH = 30
config.SIAMDW.TEST.END_EPOCH = 50

# tune
config.SIAMDW.TUNE.MODEL = config.SIAMDW.TRAIN.MODEL
config.SIAMDW.TUNE.DATA = 'VOT2015'
config.SIAMDW.TUNE.METHOD = 'GENE'  # 'GENE' or 'RAY'

# #-----————- config for siamrpn ------------
config.SIAMRPN = edict()
config.SIAMRPN.DATASET = edict()
config.SIAMRPN.DATASET.VID = edict()
config.SIAMRPN.DATASET.YTB = edict()
config.SIAMRPN.DATASET.COCO = edict()
config.SIAMRPN.DATASET.DET = edict()
config.SIAMRPN.DATASET.GOT10K = edict()
config.SIAMRPN.DATASET.LASOT = edict()
config.SIAMRPN.TRAIN = edict()
config.SIAMRPN.TEST = edict()
config.SIAMRPN.TUNE = edict()

# augmentation
config.SIAMRPN.DATASET.SHIFT = 4
config.SIAMRPN.DATASET.SCALE = 0.05
config.SIAMRPN.DATASET.COLOR = 1
config.SIAMRPN.DATASET.FLIP = 0
config.SIAMRPN.DATASET.BLUR = 0
config.SIAMRPN.DATASET.ROTATION = 0


# vid
config.SIAMRPN.DATASET.VID.PATH = '/data2/vid/crop271'
config.SIAMRPN.DATASET.VID.ANNOTATION = '/data2/vid/train.json'

# Y2B
config.SIAMRPN.DATASET.YTB.PATH = '/data2/yt_bb/crop271'
config.SIAMRPN.DATASET.YTB.ANNOTATION = '/data2/yt_bb/train.json'

# DET
config.SIAMRPN.DATASET.YTB.PATH = '/data2/det/crop271'
config.SIAMRPN.DATASET.YTB.ANNOTATION = '/data2/det/train.json'

# COCO
config.SIAMRPN.DATASET.YTB.PATH = '/data2/coco/crop271'
config.SIAMRPN.DATASET.YTB.ANNOTATION = '/data2/coco/train.json'

# GOT10K
config.SIAMRPN.DATASET.YTB.PATH = '/data2/got10k/crop271'
config.SIAMRPN.DATASET.YTB.ANNOTATION = '/data2/got10k/train.json'

# LASOT
config.SIAMRPN.DATASET.YTB.PATH = '/data2/lasot/crop271'
config.SIAMRPN.DATASET.YTB.ANNOTATION = '/data2/lasot/train.json'


# train
config.SIAMRPN.TRAIN.MODEL = "SiamRPNRes22"
config.SIAMRPN.TRAIN.RESUME = False
config.SIAMRPN.TRAIN.START_EPOCH = 0
config.SIAMRPN.TRAIN.END_EPOCH = 50
config.SIAMRPN.TRAIN.TEMPLATE_SIZE = 127
config.SIAMRPN.TRAIN.SEARCH_SIZE = 255
config.SIAMRPN.TRAIN.STRIDE = 8
config.SIAMRPN.TRAIN.BATCH = 32
config.SIAMRPN.TRAIN.PRETRAIN = 'resnet.model'
config.SIAMRPN.TRAIN.LR_POLICY = 'log'
config.SIAMRPN.TRAIN.LR = 0.01
config.SIAMRPN.TRAIN.LR_END = 0.00001
config.SIAMRPN.TRAIN.MOMENTUM = 0.9
config.SIAMRPN.TRAIN.WEIGHT_DECAY = 0.0005
config.SIAMRPN.TRAIN.CLS_WEIGHT = 1
config.SIAMRPN.TRAIN.REG_WEIGHT = 1
config.SIAMRPN.TRAIN.WHICH_USE = ['VID', 'YTB']  # VID or 'GOT10K' 
config.SIAMRPN.TRAIN.ANCHORS_RATIOS = [0.33, 0.5, 1, 2, 3]  
config.SIAMRPN.TRAIN.ANCHORS_SCALES = [8]  
config.SIAMRPN.TRAIN.ANCHORS_THR_HIGH = 0.6
config.SIAMRPN.TRAIN.ANCHORS_THR_LOW = 0.3
config.SIAMRPN.TRAIN.ANCHORS_POS_KEEP = 16    # postive anchors to calc loss
config.SIAMRPN.TRAIN.ANCHORS_ALL_KEEP = 64    # postive + neg anchors to calc loss


# test
config.SIAMRPN.TEST.MODEL = config.SIAMRPN.TRAIN.MODEL
config.SIAMRPN.TEST.DATA = 'VOT2017'
config.SIAMRPN.TEST.START_EPOCH = 30
config.SIAMRPN.TEST.END_EPOCH = 50

# tune
config.SIAMRPN.TUNE.MODEL = config.SIAMRPN.TRAIN.MODEL
config.SIAMRPN.TUNE.DATA = 'VOT2017'
config.SIAMRPN.TUNE.METHOD = 'TPE' 



def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    config[model_name][k][vk][vvk] = vvv
    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['SIAMDW', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=SIAMDW or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
