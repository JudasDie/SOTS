''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: load and save models / check trainable parameters
Data: 2021.6.23
'''
import os
import glob
import math
import torch
import torch.nn as nn
from os import makedirs
from copy import deepcopy
from os.path import join, exists
from loguru import logger
from utils.general_helper import is_parallel


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def get_latest_run(search_dir='./snapshot'):
    """
    get the latest saved ckpt (used for resume)
    :param search_dir:
    :return:
    """
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime)


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64),), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops * 100)  # 640x640 FLOPS
    except:
        fs = ''

    logger.info('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info('Using CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def change_f2b(state_dict):
    '''
    Old style model is stored with 'features.features', change it to 'backbone'
    '''
    print('change features.features to backbone')

    f = lambda x: x.replace('features.features', 'backbone') if 'features.features' in x else x
    return {f(key): value for key, value in state_dict.items()}


def addhead_clsreg(state_dict):
    '''
    'head.'+'classification'/'regression' for old style AutoMatch pretrained model
    '''

    f = lambda x: 'head.'+x if ('regression' in x or 'classification' in x) and 'head' not in x else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict, print_unuse=True):
    '''
    check keys between the pre-trained checkpoint and the model keys
    '''
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    logger.info('missing keys:{}'.format(missing_keys))
    if print_unuse:
        logger.info('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def load_pretrain(model, pretrained_path, print_unuse=True, f2b=True, addhead=False):
    '''
    load pre-trained checkpoints
    f2b: old pretrained model are saved with 'features.features'
    addhead: previous pretrained AutoMatch are loss head.+
    '''
    print('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')

    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')  # remove multi-gpu label

    if f2b:
        pretrained_dict = change_f2b(pretrained_dict)

    if addhead:
        pretrained_dict = addhead_clsreg(pretrained_dict)

    check_keys(model, pretrained_dict, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=True)
    return model


def restore_from(model, optimizer, ckpt_path):
    '''
    restir models
    '''
    print('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    arch = ckpt['arch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch, arch


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    """
    save checkpoint
    """
    torch.save(states, join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], join(output_dir, 'model_best.pth'))


def save_model(model, epoch, optimizer, model_name, cfg, isbest=False):
    """
    save model
    """
    if not exists(cfg.COMMON.CHECKPOINT_DIR):
        makedirs(cfg.COMMON.CHECKPOINT_DIR)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': model_name,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict()
    }, isbest, cfg.COMMON.CHECKPOINT_DIR, 'checkpoint_e%d.pth' % (epoch + 1))


def check_trainable(model, logger, print=True):
    """
    print trainable params
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    if print:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params

def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)
