''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: build mot models
Data: 2022.4.7
'''

import importlib
import torch.nn as nn


class simple_mot_builder(nn.Module):
    def __init__(self, opt):
        super(simple_mot_builder).__init__()
        self.opt = opt

    def build(self, pre_save_cfg=None):
        if self.opt.MODEL.Name in ['CSTrack']:
            mot_module = importlib.import_module('models.mot.cstrack')
            model_func = getattr(mot_module, 'Model')
            model = model_func(self.opt.args.cfg or pre_save_cfg, ch=3, nc=1)
        else:
            raise ValueError('not implemented tracker')

        return model



