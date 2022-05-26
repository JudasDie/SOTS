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


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransInMoHead(nn.Module):
    def __init__(self, config):
        super(TransInMoHead, self).__init__()
        in_channel = config.HEAD.IN_CHANNEL
        mid_channel = config.HEAD.MID_CHANNEL
        layer_num = config.HEAD.LAYER_NUM
        self.class_embed = MLP(in_channel, mid_channel, 2, layer_num)
        self.bbox_embed = MLP(in_channel, mid_channel, 4, layer_num)

    def forward(self, fus_feat):
        outputs_class = self.class_embed(fus_feat)
        outputs_coord = self.bbox_embed(fus_feat).sigmoid()
        out = {'cls': outputs_class[-1], 'reg': outputs_coord[-1]}
        return out


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


# SiamCAR Head
class CARHead(torch.nn.Module):
    def __init__(self, in_channels=256, num_classes=2, num_convs=4):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.xcorr_depthwise = xcorr_depthwise
        self.down = nn.ConvTranspose2d(in_channels * 3, in_channels, 1, 1)

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, xf, zf):
        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls_tower = self.cls_tower(features)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(features)))

        out = {'cls': logits, 'reg': bbox_reg, 'cen': centerness}
        return out


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
