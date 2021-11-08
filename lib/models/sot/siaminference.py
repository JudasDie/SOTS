''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: general SOT pipeline, support SiamFC, SiamDW, Ocean, AutoMatch
Data: 2021.6.23
'''

import torch
import importlib
import torch.nn as nn
import pdb


class SiamInference(nn.Module):
    def __init__(self, archs=None):
        super(SiamInference, self).__init__()
        self.cfg = archs['cfg']
        self.init_arch(archs)
        self.init_hyper()
        self.init_loss()


    def init_arch(self, inputs):
        self.backbone = inputs['backbone']
        self.neck = inputs['neck']
        self.head = inputs['head']

    def init_hyper(self):
        self.lambda_u = 0.1
        self.lambda_s = 0.2
        # self.grids()

    def init_loss(self):
        if self.cfg is None:
            raise Exception('Not set config!')

        loss_module = importlib.import_module('models.sot.loss')
        cls_loss_type = self.cfg.MODEL.LOSS.CLS_LOSS
        reg_loss_type = self.cfg.MODEL.LOSS.REG_LOSS

        self.cls_loss = getattr(loss_module, cls_loss_type)

        if self.cfg.MODEL.NAME in ['AutoMatch']:
            cls_loss_add_type = self.cfg.MODEL.LOSS.CLS_LOSS_ADDITIONAL
            self.cls_loss_additional = getattr(loss_module, cls_loss_add_type)

        if reg_loss_type is None or reg_loss_type == 'None':
            pass
        else:
            self.reg_loss = getattr(loss_module, reg_loss_type)

    def forward(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - search: BCHW, H*W:255*255
         - cls_label: BH'W' or B2H'W'
         - reg_label: B4H'W (optional)
         - reg_weight: BH'W' (optional)
        """

        template, search = inputs['template'], inputs['search']

        # backbone
        zfs = self.backbone(template)
        xfs = self.backbone(search)

        if self.cfg.MODEL.NAME in ['Ocean']:
            zf, xf = zfs['p3'], xfs['p3']
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            zf, xf = zfs['p2'], xfs['p2']
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            zf_conv4, zf_conv3 = zfs['p3'], zfs['p2']
            xf_conv4, xf_conv3 = xfs['p3'], xfs['p2']
        else:
            raise Exception('Not implemented model!')

        # neck
        if self.neck is not None:
            if self.cfg.MODEL.NAME in ['Ocean']:
                xf_neck = self.neck(xf, crop=False)
                zf_neck = self.neck(zf, crop=True)
                zf, xf = zf_neck['crop'], xf_neck['ori']
            elif self.cfg.MODEL.NAME in ['AutoMatch']:
                zfs4, zfs3 = self.neck(zf_conv4, zf_conv3)
                xfs4, xfs3 = self.neck(xf_conv4, xf_conv3)

        # head
        # not implement Ocean object-aware version, if you need, pls find it in researchmm/TracKit
        if self.cfg.MODEL.NAME in ['AutoMatch']:
            head_inputs = {'xf_conv4': xfs4, 'xf_conv3': xfs3, 'zf_conv4': zfs4, 'zf_conv3': zfs3, \
                          'template_mask': inputs['template_mask'], 'target_box': inputs['template_bbox'],
                           'jitterBox': inputs['jitterBox'], 'cls_label': inputs['cls_label']
                           }

            cls_preds, reg_preds = self.head(head_inputs)
        else:
            preds = self.head(xf, zf)

        if self.cfg.MODEL.NAME in ['Ocean']:
            cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
            cls_pred, reg_pred = preds['cls'], preds['reg']

            reg_loss = self.reg_loss(reg_pred, reg_label, reg_weight)
            cls_loss = self.cls_loss(cls_pred, cls_label)
            loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss}
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
            reg_pred = reg_preds['reg_score']
            reg_loss = 2 * self.reg_loss(reg_pred, reg_label, reg_weight)
            cls_pred_s1, cls_pred_s2 = cls_preds['cls_score_s1'], cls_preds['cls_score_s2']
            cls_loss_s1 = self.cls_loss(cls_pred_s1, cls_label)
            cls_loss_s2 = self.cls_loss_additional(cls_pred_s2, cls_preds['cls_label_s2'], cls_preds['cls_jitter'], inputs['jitter_ious'])
            cls_loss = cls_loss_s1 + cls_loss_s2
            loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss}
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            cls_label = inputs['cls_label']
            cls_pred = preds['cls']

            cls_loss = self.cls_loss(cls_pred, cls_label)
            loss = {'cls_loss': cls_loss}
        else:
            raise Exception('not supported model')

        return loss

    # only for testing
    def template(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - template_mask: BHW (optional)
        """

        template = inputs['template']

        zfs = self.backbone(template)

        if self.cfg.MODEL.NAME in ['Ocean']:
            zf = zfs['p3']
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            zf = zfs['p2']
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            zf_conv4, zf_conv3 = zfs['p3'], zfs['p2']
        else:
            raise Exception('Not implemented model!')

        if self.neck is not None:
            if self.cfg.MODEL.NAME in ['Ocean']:
                zf_neck = self.neck(zf, crop=True)
                self.zf = zf_neck['crop']
            elif self.cfg.MODEL.NAME in ['AutoMatch']:
                self.zfs4, self.zfs3 = self.neck(zf_conv4, zf_conv3)

        else:
            self.zf = zf

        if 'template_mask' in inputs.keys():
            self.template_mask = inputs['template_mask'].float()

        if 'target_box' in inputs.keys():
            self.target_box = torch.tensor(inputs['target_box'], dtype=torch.float32).to(self.zfs3.device)
            self.target_box = self.target_box.view(1, 4)

        if self.cfg.MODEL.NAME in ['OceanPlus']:  # update template
            self.MA_kernel = self.zf.detach()
            self.zf_update = None

    def track(self, inputs):
        """
        inputs:
         - search: BCHW, H*W:255*255
        """

        search = inputs['search']
        xfs = self.backbone(search)

        if self.cfg.MODEL.NAME in ['Ocean']:
            xf = xfs['p3']
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            xf = xfs['p2']
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            xf_conv4, xf_conv3 = xfs['p3'], xfs['p2']
        else:
            raise Exception('Not implemented model!')

        if self.neck is not None:
            if self.cfg.MODEL.NAME in ['AutoMatch']:
                xfs4, xfs3 = self.neck(xf_conv4, xf_conv3)
                head_inputs = {'xf_conv4': xfs4, 'xf_conv3': xfs3, 'zf_conv4': self.zfs4, 'zf_conv3': self.zfs3, \
                                'template_mask': self.template_mask, 'target_box': self.target_box, }

                cls_preds, reg_preds = self.head(head_inputs)
                preds = {
                    'cls_s1': cls_preds['cls_score_s1'],
                    'cls_s2': cls_preds['cls_score_s2'],
                    'reg': reg_preds['reg_score'] # clip large regression pred
                }

                # record some feats for zoom
                self.record = [cls_preds['xf_conv4'].detach(), cls_preds['xf_conv3'].detach(),
                               cls_preds['zf_conv4'].detach(), cls_preds['zf_conv3'].detach()]  # [xf_conv4, xf_conv3, zf_conv4, zf_conv3]
            else:
                xf_neck = self.neck(xf, crop=False)
                xf = xf_neck['ori']
                preds = self.head(xf, self.zf)

        else:
            preds = self.head(xf, self.zf)

        if 'reg' not in preds.keys():
            preds['reg'] = None

        return preds

    def zoom(self, box):
        """
        zoom trick in AutoMatch
        """
        cls_pred = self.head.classification(None, self.record[0], self.record[2], self.record[1], self.record[3], zoom_box=box)

        return cls_pred.squeeze()













