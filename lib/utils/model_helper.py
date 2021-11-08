''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: load and save models / check trainable parameters
Data: 2021.6.23
'''

import math
import torch
from os import makedirs
from os.path import join, exists

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

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
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
    model.load_state_dict(pretrained_dict, strict=False)
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
