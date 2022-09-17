''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: general SOT pipeline, support SiamFC, SiamDW, Ocean, AutoMatch
Data: 2021.6.23
'''

import numpy as np
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
import pdb

from utils.box_helper import matcher
from models.sot.modules import NestedTensor, nested_tensor_from_tensor, nested_tensor_from_tensor_2


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
        if self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
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
        elif self.cfg.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
            nas_list_z = inputs['nas_list_z']
            nas_list_x = inputs['nas_list_x']
            nas_list_nlp = inputs['nas_list_nlp']

            label_loc_tem = inputs['template_bbox'].cuda()
            if inputs['phrase_ids'] is None:  # this is impossible, for train cat in pytorch
                phrase = [torch.zeros([template.size(0), 50], dtype=torch.long).cuda(),
                          torch.zeros([template.size(0), 50], dtype=torch.long).cuda()]
            else:
                phrase = [inputs['phrase_ids'].cuda(), inputs['phrase_attnmask'].cuda()]
            if nas_list_z is None or nas_list_z == 'None':
                nas_list_z = [np.random.randint(4) for i in range(20)]
                nas_list_x = [np.random.randint(4) for i in range(20)]
            if None in nas_list_nlp:
                nas_list_nlp = [[np.random.randint(4) for i in range(4)], [np.random.randint(4) for i in range(4)]]
            if self.cfg.MODEL.NAME in ['VLT_SCAR']:
                if phrase[0].max() == 0 and phrase[0].min() == 0:
                    zfs = self.backbone.forward_track_nlp_tem(template, nas_list_z, None, nas_list_nlp[0],
                                                              batch_box=label_loc_tem)
                    xfs = self.backbone.forward_track_nlp_sear(search, nas_list_x, None, nas_list_nlp[1])
                else:
                    nlp_tokens = self.backbone.forward_nlp(phrase)
                    zfs = self.backbone.forward_track_nlp(template, nas_list_z, nlp_tokens, nas_list_nlp[0])
                    xfs = self.backbone.forward_track_nlp(search, nas_list_x, nlp_tokens, nas_list_nlp[1])
            elif self.cfg.MODEL.NAME in ['VLT_TT']:
                if phrase[0].max() == 0 and phrase[0].min() == 0:
                    vistoken = True
                    nlp_tokens = None
                else:
                    vistoken = False
                    nlp_tokens = self.backbone[0].body.forward_nlp(phrase)
                src_z, pos_z = self.backbone(template, nas_lists=[nas_list_z, nas_list_x], nlp_tokens=nlp_tokens, vistoken=vistoken, batch_box=label_loc_tem, nlp_cand=nas_list_nlp[0])
                src_x, pos_x = self.backbone(search, nas_lists=[nas_list_z, nas_list_x], nlp_tokens=nlp_tokens, vistoken=vistoken, nlp_cand=nas_list_nlp[1])
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
        elif self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
            zf = zfs
            xf = xfs
        elif self.cfg.MODEL.NAME in ['VLT_TT']:
            mask_x, mask_z = None, None
        # elif self.cfg.MODEL.NAME in ['Ocean_nasnlp']:
        #     zf, xf = zfs[-1], xfs[-1]
        #     if zf.size(-1) != 15:
        #         zf = F.interpolate(zf, size=[15, 15], mode='bicubic', align_corners=False)
        #     if xf.size(-1) != 31:
        #         xf = F.interpolate(xf, size=[31, 31], mode='bicubic', align_corners=False)
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
            elif self.cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
                fused_zx = self.neck(src_z, mask_z, src_x, mask_x, pos_z[-1], pos_x[-1])
            elif self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
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
        elif self.cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
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
        elif self.cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            cls_label, search_bbox = inputs['cls_label'], inputs['search_bbox']
            cls_pred, reg_pred = preds['cls'], preds['reg']

            search_bbox = search_bbox / self.cfg.TRAIN.SEARCH_SIZE
            indices = matcher(reg_pred, search_bbox)
            reg_loss = self.reg_loss(reg_pred, search_bbox, indices)
            cls_loss = self.cls_loss(cls_pred, cls_label, indices, self.cfg)
            loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss}
        elif self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
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
        elif self.cfg.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
            self.nas_list_z = inputs['nas_list_z']
            self.nas_list_x = inputs['nas_list_x']
            self.nas_list_nlp = inputs['nas_list_nlp']
            label_loc_tem = inputs['template_bbox'].cuda()
            phrase = inputs['phrase']
            if phrase == [] or phrase is None:
                # print('reset')
                phrase = [torch.zeros([1, 50], dtype=torch.long).cuda(), torch.zeros([1, 50], dtype=torch.long).cuda()]
                # self.phrase = None
            else:
                if self.cfg.MODEL.NAME in ['VLT_SCAR']:
                    phrase = self.backbone.bert_tokenizer.batch_encode_plus(phrase, padding='longest', return_tensors='pt')
                elif self.cfg.MODEL.NAME in ['VLT_TT']:
                    phrase = self.backbone[0].body.bert_tokenizer.batch_encode_plus(phrase, padding='longest',
                                                                            return_tensors='pt')
                phrase_ids = phrase['input_ids'].squeeze()
                phrase_ids = torch.cat([phrase_ids, torch.zeros(50 - phrase_ids.size(0), dtype=torch.long)], dim=0)
                phrase_attnmask = phrase['attention_mask'].squeeze()
                phrase_attnmask = torch.cat(
                    [phrase_attnmask, torch.zeros(50 - phrase_attnmask.size(0), dtype=torch.long)],
                    dim=0)
                phrase = [phrase_ids.cuda().unsqueeze(0), phrase_attnmask.cuda().unsqueeze(0)]
            if self.cfg.MODEL.NAME in ['VLT_SCAR']:
                if phrase[0].max() == 0 and phrase[0].min() == 0:
                    self.phrase = None
                    zfs = self.backbone.forward_track_nlp_tem(template, self.nas_list_z, None, self.nas_list_nlp[0],
                                                              batch_box=label_loc_tem)
                else:
                    self.phrase = self.backbone.forward_nlp(phrase)
                    zfs = self.backbone.forward_track_nlp(template, self.nas_list_z, self.phrase, self.nas_list_nlp[0])
            elif self.cfg.MODEL.NAME in ['VLT_TT']:
                if phrase[0].max() == 0 and phrase[0].min() == 0:
                    self.vistoken = True
                    self.phrase = None
                else:
                    self.vistoken = False
                    self.phrase = self.backbone[0].body.forward_nlp(phrase)
                src_z, pos_z = self.backbone(template, nas_lists=[self.nas_list_z, self.nas_list_x], nlp_tokens=self.phrase,
                                             vistoken=self.vistoken, batch_box=label_loc_tem, nlp_cand=self.nas_list_nlp[0])
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
        elif self.cfg.MODEL.NAME in ['VLT_SCAR']:
            zf = zfs
            for i in range(len(zf)):
                if zf[i].size(-1) != 15:
                    zf[i] = F.interpolate(zf[i], size=[15, 15], mode='bicubic', align_corners=False)
        elif self.cfg.MODEL.NAME in ['VLT_TT']:
            mask_z = None
        else:
            raise Exception('Not implemented model!')

        if self.neck is not None:
            if self.cfg.MODEL.NAME in ['Ocean']:
                zf_neck = self.neck(zf, crop=True)
                self.zf = zf_neck['crop']
            elif self.cfg.MODEL.NAME in ['AutoMatch']:
                self.zfs4, self.zfs3 = self.neck(zf_conv4, zf_conv3)
            elif self.cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
                self.src_z, self.mask_z, self.pos_z = src_z, mask_z, pos_z
            elif self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
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
        elif self.cfg.MODEL.NAME in ['VLT_SCAR']:
            if self.phrase is None:
                xfs = self.backbone.forward_track_nlp_sear(search, self.nas_list_x, None, self.nas_list_nlp[1])
            else:
                xfs = self.backbone.forward_track_nlp(search, self.nas_list_x, self.phrase, self.nas_list_nlp[1])
        elif self.cfg.MODEL.NAME in ['VLT_TT']:
            src_x, pos_x = self.backbone(search, nas_lists=[self.nas_list_z, self.nas_list_x], nlp_tokens=self.phrase,
                                         vistoken=self.vistoken, nlp_cand=self.nas_list_nlp[1])
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
        elif self.cfg.MODEL.NAME in ['VLT_SCAR']:
            xf = xfs
            for i in range(len(xf)):
                if xf[i].size(-1) != 31:
                    xf[i] = F.interpolate(xf[i], size=[31, 31], mode='bicubic', align_corners=False)
        elif self.cfg.MODEL.NAME in ['VLT_TT']:
            mask_x = None
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
            elif self.cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
                fused_zx = self.neck(self.src_z, self.mask_z, src_x, mask_x, self.pos_z[-1], pos_x[-1])
                preds = self.head(fused_zx)
            elif self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
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













