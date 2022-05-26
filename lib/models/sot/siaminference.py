''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: general SOT pipeline, support SiamFC, SiamDW, Ocean, AutoMatch
Data: 2021.6.23
'''

import torch
import importlib
import torch.nn as nn
import pdb

from lib.utils.box_helper import matcher
from lib.models.sot.modules import NestedTensor, nested_tensor_from_tensor, nested_tensor_from_tensor_2


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
        if self.cfg.MODEL.NAME in ['CNNInMo']:
            loss_type = getattr(loss_module, self.cfg.MODEL.LOSS.LOSS_FUNC)
            self.loss_func = loss_type(output_size=self.cfg.MODEL.LOSS.OUTPUT_SIZE)
        else:
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
        if self.cfg.MODEL.NAME in ['TransInMo']:
            if not isinstance(template, NestedTensor):
                template = nested_tensor_from_tensor(template)
            if not isinstance(search, NestedTensor):
                search = nested_tensor_from_tensor(search)
            zf, pos_z = self.backbone(template)
            xf, pos_x = self.backbone(search)
        else:
            zfs = self.backbone(template)
            xfs = self.backbone(search)

        if self.cfg.MODEL.NAME in ['Ocean']:
            zf, xf = zfs['p3'], xfs['p3']
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            zf, xf = zfs['p2'], xfs['p2']
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            zf_conv4, zf_conv3 = zfs['p3'], zfs['p2']
            xf_conv4, xf_conv3 = xfs['p3'], xfs['p2']
        elif self.cfg.MODEL.NAME in ['TransInMo']:
            src_x, mask_x = xf[-1].decompose()
            assert mask_x is not None
            src_z, mask_z = zf[-1].decompose()
            assert mask_z is not None
        elif self.cfg.MODEL.NAME in ['CNNInMo']:
            zf = zfs
            xf = xfs
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
            elif self.cfg.MODEL.NAME in ['TransInMo']:
                fused_zx = self.neck(src_z, mask_z, src_x, mask_x, pos_z[-1], pos_x[-1])
            elif self.cfg.MODEL.NAME in ['CNNInMo']:
                zf = self.neck(zf)
                xf = self.neck(xf)

        # head
        # not implement Ocean object-aware version, if you need, pls find it in researchmm/TracKit
        if self.cfg.MODEL.NAME in ['AutoMatch']:
            head_inputs = {'xf_conv4': xfs4, 'xf_conv3': xfs3, 'zf_conv4': zfs4, 'zf_conv3': zfs3, \
                          'template_mask': inputs['template_mask'], 'target_box': inputs['template_bbox'],
                           'jitterBox': inputs['jitterBox'], 'cls_label': inputs['cls_label']
                           }

            cls_preds, reg_preds = self.head(head_inputs)
        elif self.cfg.MODEL.NAME in ['TransInMo']:
            preds = self.head(fused_zx)
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
        elif self.cfg.MODEL.NAME in ['TransInMo']:
            cls_label, search_bbox = inputs['cls_label'], inputs['search_bbox']
            cls_pred, reg_pred = preds['cls'], preds['reg']

            search_bbox = search_bbox / self.cfg.TRAIN.SEARCH_SIZE
            indices = matcher(reg_pred, search_bbox)
            reg_loss = self.reg_loss(reg_pred, search_bbox, indices)
            cls_loss = self.cls_loss(cls_pred, cls_label, indices, self.cfg)
            loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss}
        elif self.cfg.MODEL.NAME in ['CNNInMo']:
            cls_label, search_bbox = inputs['cls_label'], inputs['search_bbox']
            cls_loss, reg_loss, cen_loss = self.loss_func(preds, cls_label, search_bbox)
            loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'cen_loss': cen_loss}
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

        if self.cfg.MODEL.NAME in ['TransInMo']:
            if not isinstance(template, NestedTensor):
                template = nested_tensor_from_tensor_2(template)
            zf, pos_z = self.backbone(template)
        else:
            zfs = self.backbone(template)

        if self.cfg.MODEL.NAME in ['Ocean']:
            zf = zfs['p3']
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            zf = zfs['p2']
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            zf_conv4, zf_conv3 = zfs['p3'], zfs['p2']
        elif self.cfg.MODEL.NAME in ['TransInMo']:
            src_z, mask_z = zf[-1].decompose()
            assert mask_z is not None
        elif self.cfg.MODEL.NAME in ['CNNInMo']:
            zf = zfs
        else:
            raise Exception('Not implemented model!')

        if self.neck is not None:
            if self.cfg.MODEL.NAME in ['Ocean']:
                zf_neck = self.neck(zf, crop=True)
                self.zf = zf_neck['crop']
            elif self.cfg.MODEL.NAME in ['AutoMatch']:
                self.zfs4, self.zfs3 = self.neck(zf_conv4, zf_conv3)
            elif self.cfg.MODEL.NAME in ['TransInMo']:
                self.src_z, self.mask_z, self.pos_z = src_z, mask_z, pos_z
            elif self.cfg.MODEL.NAME in ['CNNInMo']:
                self.zf = self.neck(zf)
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
        if self.cfg.MODEL.NAME in ['TransInMo']:
            if not isinstance(search, NestedTensor):
                search = nested_tensor_from_tensor_2(search)
            xf, pos_x = self.backbone(search)
        else:
            xfs = self.backbone(search)

        if self.cfg.MODEL.NAME in ['Ocean']:
            xf = xfs['p3']
        elif self.cfg.MODEL.NAME in ['SiamDW']:
            xf = xfs['p2']
        elif self.cfg.MODEL.NAME in ['AutoMatch']:
            xf_conv4, xf_conv3 = xfs['p3'], xfs['p2']
        elif self.cfg.MODEL.NAME in ['TransInMo']:
            src_x, mask_x = xf[-1].decompose()
            assert mask_x is not None
        elif self.cfg.MODEL.NAME in ['CNNInMo']:
            xf = xfs
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
            elif self.cfg.MODEL.NAME in ['TransInMo']:
                fused_zx = self.neck(self.src_z, self.mask_z, src_x, mask_x, self.pos_z[-1], pos_x[-1])
                preds = self.head(fused_zx)
            elif self.cfg.MODEL.NAME in ['CNNInMo']:
                xf = self.neck(xf)
                preds = self.head(xf, self.zf)
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













