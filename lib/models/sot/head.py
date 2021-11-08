''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: neck modules for SOT models
Data: 2021.6.23
'''

import torch
import torch.nn as nn
import numpy as np
from .modules import *

class Learn2Match(nn.Module):
    """
    target estimation head in "learn to match: Learn to Match: Automatic Ma tching Networks Design for Visual Tracking"
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """
    def __init__(self, in_channels=256, out_channels=256, roi_size=3):
        super(Learn2Match, self).__init__()
        # default parameters
        self.search_size = 255
        self.score_size = (self.search_size - 255) // 8 + 31
        self.batch = 32 if self.training else 1
        self.grids()

        # heads
        self.regression = L2Mregression(inchannels=in_channels, outchannels=out_channels, towernum=3)
        self.classification = L2Mclassification(roi_size=roi_size, stride=8.0, inchannels=in_channels)

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        stride = 8

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

        self.grid_to_search_x.requires_grad = False
        self.grid_to_search_y.requires_grad = False

    def pred_to_image(self, bbox_pred):
        if not bbox_pred.size(0) == self.batch:
            self.batch = bbox_pred.size(0)
            self.grids()

        if not self.score_size == bbox_pred.size(-1):
            self.score_size = bbox_pred.size(-1)
            self.grids()

        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def forward(self, inputs):
        xfs4, xfs3, zfs4, zfs3, template_mask, target_box,  = inputs['xf_conv4'], inputs['xf_conv3'], inputs['zf_conv4'], \
                                                              inputs['zf_conv3'], inputs['template_mask'], inputs['target_box']

        reg_outputs = self.regression(xf=xfs4, zf=zfs4, zfs3=zfs3, mask=template_mask, target_box=target_box)
        
        pred_box, target = self.pred_to_image(reg_outputs['reg_score']), [reg_outputs['zf_conv4'], reg_outputs['zf_conv3']]

        if self.training:
            cls_label, jitterBox = inputs['cls_label'], inputs['jitterBox']
        else:
            cls_label, jitterBox = None, None
        cls_outputs = self.classification(pred_box, reg_outputs['reg_feature'], zfs4, xfs3, zfs3, target=target,
                                          mask=template_mask, cls_label=cls_label, jitterBox=jitterBox)

        return cls_outputs, reg_outputs


class FCOSOcean(nn.Module):
    """
    FCOS like regression (Ocean)
    """
    def __init__(self, in_channels=256, out_channels=256, towernum=4, align=False):
        super(FCOSOcean, self).__init__()
        bbox_tower = []  # for regression
        cls_tower = []   # for classification

        # multi-dilation encoding base
        self.cls_encode = matrix(in_channels=in_channels, out_channels=out_channels)
        self.reg_encode = matrix(in_channels=in_channels, out_channels=out_channels)
        self.cls_dw = GroupDW()
        self.reg_dw = GroupDW()

        # regression head
        for i in range(towernum):
            if i == 0:
                bbox_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            else:
                bbox_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

            bbox_tower.append(nn.BatchNorm2d(out_channels))
            bbox_tower.append(nn.ReLU())

        # classification head
        for i in range(towernum):
            if i == 0:
                cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

            cls_tower.append(nn.BatchNorm2d(out_channels))
            cls_tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        # prediction layers
        self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

        # object-aware align
        if align:
            raise Exception('Not implemented align module in this repo. Please refer to researchmm/TracKit. \
                            It equals to ROIAlign, please refer to AutoMatch')

    def forward(self, search, kernal, update=None):
        # encode first
        if update is None:
            cls_z, cls_x = self.cls_encode(kernal, search)   # [z11, z12, z13]
        else:
            cls_z, cls_x = self.cls_encode(update, search)  # [z11, z12, z13]

        reg_z, reg_x = self.reg_encode(kernal, search)  # [x11, x12, x13]

        # matching learning
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)

        # predictions
        x_reg = self.bbox_tower(reg_dw)
        r = self.adjust * self.bbox_pred(x_reg) + self.bias
        reg = torch.exp(r)

        x_cls = self.cls_tower(cls_dw)
        cls = 0.1 * self.cls_pred(x_cls)

        return {'reg_feature': x_reg, 'cls_feature': x_cls, 'reg': reg, 'cls': cls}


class SiamFCCorr(nn.Module):
    """
    original cross-correlation head used in SiamFC, SiamDW
    """
    def __init__(self):
        super(SiamFCCorr, self).__init__()

    def _conv2d_group(self, x, kernel):
        batch = x.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, x_f, z_f):
        if not self.training:
            cls = 0.1 * F.conv2d(x_f, z_f)
            return {'cls': cls}
        else:
            cls = 0.1 * self._conv2d_group(x_f, z_f)
            return {'cls': cls}
