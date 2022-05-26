''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: modules for SOT models
Data: 2021.6.23
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import Dict, List, Optional
from .match import *


def xcorr_depthwise(x, kernel):
    '''
    depthwise cross correlation
    SiamRPN++: https://arxiv.org/abs/1812.11703
    '''

    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


# ---------------------------------------------------------------------------
# Moduels in InMo [IJCAI2022]: https://arxiv.org/pdf/2201.02526.pdf
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Moduels in Ocean [ECCV2020]: https://arxiv.org/abs/2006.10721
# ---------------------------------------------------------------------------

class SingleDW(nn.Module):
    '''
    a simple depthwise cross correlation layer
    '''
    def __init__(self):
        super(SingleDW, self).__init__()

    def forward(self, z, x):

        s = xcorr_depthwise(x, z)

        return s


class GroupDW(nn.Module):
    '''
    parallel depthwise cross correlation layers
    Ocean: https://arxiv.org/abs/2006.10721
    '''
    def __init__(self):
        super(GroupDW, self).__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, z, x):
        z11, z12, z21 = z
        x11, x12, x21 = x

        re11 = xcorr_depthwise(x11, z11)
        re12 = xcorr_depthwise(x12, z12)
        re21 = xcorr_depthwise(x21, z21)
        re = [re11, re12, re21]

        # weight
        weight = F.softmax(self.weight, 0)

        s = 0
        for i in range(3):
            s += weight[i] * re[i]

        return s


class matrix(nn.Module):
    """
    parallel multidilation encoding
    Ocean: https://arxiv.org/abs/2006.10721
    """
    def __init__(self, in_channels, out_channels):
        super(matrix, self).__init__()

        # same size (11)
        self.matrix11_k = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix11_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # h/2, w
        self.matrix12_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix12_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # w/2, h
        self.matrix21_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix21_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, x):
        z11 = self.matrix11_k(z)
        x11 = self.matrix11_s(x)

        z12 = self.matrix12_k(z)
        x12 = self.matrix12_s(x)

        z21 = self.matrix21_k(z)
        x21 = self.matrix21_s(x)

        return [z11, z12, z21], [x11, x12, x21]


# class AdaptiveConv(nn.Module):
#     """ Adaptive Conv is built based on Deformable Conv
#     with precomputed offsets which derived from anchors"""
#
#     def __init__(self, in_channels, out_channels):
#         super(AdaptiveConv, self).__init__()
#         self.conv = DeformConv(in_channels, out_channels, 3, padding=1)
#
#     def forward(self, x, offset):
#         N, _, H, W = x.shape
#         assert offset is not None
#         assert H * W == offset.shape[1]
#         # reshape [N, NA, 18] to (N, 18, H, W)
#         offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
#         x = self.conv(x, offset)
#
#         return x


# class AlignHead(nn.Module):
#     """
#     feature alignment module
#     Ocean: https://arxiv.org/abs/2006.10721
#     """
#
#     def __init__(self, in_channels, feat_channels):
#         super(AlignHead, self).__init__()
#
#         self.rpn_conv = AdaptiveConv(in_channels, feat_channels)
#         self.rpn_cls = nn.Conv2d(feat_channels, 1, 1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, offset):
#         x = self.relu(self.rpn_conv(x, offset))
#         cls_score = self.rpn_cls(x)
#         return cls_score


# ---------------------------------------------------------------------------
# Moduels in AutoMatch [ICCV2021]:
# https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
# ---------------------------------------------------------------------------

class L2Mregression(nn.Module):
    """
    bounding box regression head in AutoMatch
    "Learn to Match: Automatic Matching Networks Design for Visual Tracking"
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, inchannels=512, outchannels=256, towernum=3):
        super(L2Mregression, self).__init__()

        # encode features
        self.reg_encode = SimpleMatrix(in_channels=inchannels, out_channels=64)

        # learned matching network
        roi_size, stride = 3, 8
        self.roi_te = roi_template(roi_size=roi_size, stride=stride, inchannels=inchannels)  # template to 1*1
        self.LTM = LTM(inchannel=256, used=[5, 6])


        # box pred head
        tower = []
        for i in range(towernum):
            if i == 0:
                tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            else:
                tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))

            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())


    def forward(self, xf, zf, zfs3=None, mask=None, target_box=None):
        """
        Args:
            xf: conv4 search feature
            zf: conv4 template feature
            zfs3: conv3 template feature
            mask: mask of template
            target box: bounding box in template
        """

        # get roi feature of template (conv3 and conv4)
        ts4, ts3 = self.roi_te(target_box, zf, zfs3)

        # matching
        xf_ltm = self.LTM(xf, zf, ts4, mask)

        # bounding box prediction
        x_reg = self.bbox_tower(xf_ltm)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)
        if self.training:
            x = torch.clamp(x, 0, 255)
        outputs = {
            'reg_score': x,
            'reg_feature': x_reg,
            'zf_conv4': ts4,
            'zf_conv3': ts3
        }

        return outputs


class L2Mclassification(nn.Module):
    """
    bounding box regression head in AutoMatch
    "Learn to Match: Automatic Matching Networks Design for Visual Tracking"
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, roi_size=3, stride=8.0, inchannels=256):
        super(L2Mclassification, self).__init__()

        self.LTM = LTM(inchannel=256, used=[4, 5])   # matching
        self.roi_cls = roi_classification(roi_size=roi_size, stride=stride, inchannels=inchannels)


    def forward(self, pred_box, xf, zf, xfs3, zfs3, target=None, mask=None, cls_label=None, jitterBox=None,
                zoom_box=None):

        """
        Args:
            pred_box: predicted bounding boxes from the regression network
            xf: conv4 search feature
            zf: conv4 template feature
            xfs3: conv3 search feature
            zfs3: conv3 template feature
            target: temoplate feature from the output of regression network
            mask: template mask
            cls_label: label for the classfication task
            jitterBox: random select several boxes from the GT box of search image to get more positive (and hard) box
            zoombox: a small trcik during testing, optional (almost no gains)
        """

        if zoom_box is not None:
            cls_pred = self.roi_cls(None, xf, xfs3, zf, zfs3, zoom_box=zoom_box)
            return cls_pred

        # use Conv3 in classification
        ts4, ts3 = target
        xfs3 = self.LTM(xfs3, zfs3, ts3, mask)

        if self.training:
            outputs = self.roi_cls(pred_box, xf, xfs3, ts4, ts3, cls_label=cls_label, jitterBox=jitterBox)
            return outputs
        else:
            outputs = self.roi_cls(pred_box, xf, xfs3, ts4, ts3)
            return outputs


class roi_classification(nn.Module):
    """
    subclass of the classification network
    """

    def __init__(self, roi_size=3, stride=8.0, inchannels=256, alpha=0.1):
        """
        Args:
            roi_size: output size of roi
            stride: network stride
            inchannels: input channels
            alpha: for leaky-relu
        """
        super(roi_classification, self).__init__()
        self.roi_size = roi_size
        self.stride = float(stride)
        self.inchannels = inchannels

        self.fea_encoder = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
            nn.Conv2d(inchannels, inchannels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels // 2),
            nn.LeakyReLU(alpha),
        )

        self.fea_encoder_s3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
            nn.Conv2d(inchannels, inchannels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels // 2),
            nn.LeakyReLU(alpha),
        )

        self.down_spatial_conv = nn.Sequential(
            nn.Conv2d(inchannels // 2, inchannels // 2, kernel_size=roi_size, stride=1),
            nn.BatchNorm2d(inchannels // 2),
            nn.LeakyReLU(alpha),
        )

        self.down_spatial_linear = nn.Sequential(
            nn.Linear(inchannels // 2, inchannels // 2),
            nn.LeakyReLU(alpha)
        )

        self.down_spatial_conv_s3 = nn.Sequential(
            nn.Conv2d(inchannels // 2, inchannels // 2, kernel_size=roi_size, stride=1),
            nn.BatchNorm2d(inchannels // 2),
            nn.LeakyReLU(alpha),
        )

        self.down_target_s3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(inchannels // 2),
            nn.LeakyReLU(alpha),
        )

        self.down_target_s4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(inchannels // 2),
            nn.LeakyReLU(alpha),
        )

        self.down_spatial_linear_s3 = nn.Sequential(
            nn.Linear(inchannels // 2, inchannels // 2),
            nn.LeakyReLU(alpha)
        )

        self.merge_s3s4_s2 = nn.Sequential(
            nn.Linear(inchannels, inchannels),
            nn.LeakyReLU(alpha),
            nn.Linear(inchannels, inchannels),
            nn.LeakyReLU(alpha),
        )

        self.merge_s3s4_s1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
        )

        self.pred_s1 = nn.Conv2d(inchannels, 1, kernel_size=3, stride=1, padding=1)
        self.pred_s2 = nn.Linear(inchannels, 1)

        self.box_indices = None
        self.zoom_indices = None
        self.jitter_indices = None

        self.size = 31
        self.batch = 32
        batch_index = torch.Tensor([self.size * [i] for i in range(self.batch)])
        self.batch_index = batch_index.view(-1)
        self.tail_index = torch.Tensor(list(range(self.size)) * self.batch)

    def obtain_max_index(self, score, bbox, cls_label):
        """
        :param score: [B, H, W]
        :param bbox: [B, 4, H, W]
        :return:
        """
        score = score.squeeze()
        # B, H, W = score.size()

        # row
        _, indicesR = torch.max(score, 1)
        indicesR = indicesR.view(-1)

        batch_index = self.batch_index.to(indicesR.device)
        tail_index = self.tail_index.to(indicesR.device)

        indicesR = torch.stack((batch_index.float(), indicesR.float(), tail_index.float()), dim=-1)

        _, indicesC = torch.max(score, 2)
        indicesC = indicesC.view(-1)
        indicesC = torch.stack((batch_index.float(), tail_index.float(), indicesC.float()), dim=-1)
        indicesR, indicesC = indicesR.long(), indicesC.long()

        # score[indicesR[:, 0].long(), indicesR[:, 1].long(), indicesR[:, 2].long()].shape
        # sampling
        bbox_selectR = bbox[indicesR[:, 0], :, indicesR[:, 1], indicesR[:, 2]]
        bbox_selectC = bbox[indicesC[:, 0], :, indicesC[:, 1], indicesC[:, 2]]
        bbox_select = torch.cat((bbox_selectR, bbox_selectC), dim=0)

        cls_labelR = cls_label[indicesR[:, 0], indicesR[:, 1], indicesR[:, 2]]
        cls_labelC = cls_label[indicesC[:, 0], indicesC[:, 1], indicesC[:, 2]]
        cls_label = torch.cat((cls_labelR, cls_labelC), dim=0)

        return bbox_select, cls_label

    def forward(self, boxes, fea, feas3, target_s4, target_s3, cls_label=None, jitterBox=None, zoom_box=None):
        """
        Args:
            boxes: [b, 4, h, w]   predicted box
            fea: [b, c, h, w]     search feature
            target_fea: [b, c]    1*1 template feature

        Returns: cls results
        """
        if zoom_box is not None:  # optional: only used during testing
            zoom_box = zoom_box.to(fea.device)
            # -------- zoom refine -------
            if self.zoom_indices is None or not zoom_box.shape[0] == self.zoom_indices.size()[0]:
                zoom_indices = []
                zoom_indices.extend([0] * zoom_box.shape[0])
                self.zoom_indices = torch.tensor(zoom_indices, dtype=torch.float32)
                self.zoom_indices = self.zoom_indices.unsqueeze(-1).to(fea.device)

            zoomBox = torch.cat((self.zoom_indices, zoom_box), dim=1).float()

            pool_fea_zoom = roi_align(fea, zoomBox, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                      sampling_ratio=-1)  # [K, C 3, 3]
            pool_fea_s3_zoom = roi_align(feas3, zoomBox, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                         sampling_ratio=-1)  # [K, C 3, 3]

            pool_fea_zoom = self.down_spatial_linear(self.down_spatial_conv(pool_fea_zoom).squeeze())  # [K, C]
            pool_fea_s3_zoom = self.down_spatial_linear_s3(self.down_spatial_conv_s3(pool_fea_s3_zoom).squeeze())  # [K, C]

            pool_fea_zoom = pool_fea_zoom * target_s4
            pool_fea_s3_zoom = pool_fea_s3_zoom * target_s3
            pool_fea_zoom_merge = torch.cat((pool_fea_zoom, pool_fea_s3_zoom), dim=-1)
            pool_fea_zoom_merge = self.merge_s3s4_s2(pool_fea_zoom_merge)
            cls_zoom = self.pred_s2(pool_fea_zoom_merge)

            return cls_zoom

        B, _, H, W = boxes.size()

        # --- stage1 cls --- (no pooling with background)
        fea = self.fea_encoder(fea)
        feas3 = self.fea_encoder_s3(feas3)
        target_s4 = self.down_target_s4(target_s4)
        target_s3 = self.down_target_s3(target_s3)
        pool_fea = fea * target_s4
        pool_fea_s3 = feas3 * target_s3
        infuse_fea = torch.cat((pool_fea, pool_fea_s3), dim=1)
        infuse_fea = self.merge_s3s4_s1(infuse_fea)
        cls_s1 = self.pred_s1(infuse_fea)


        # --- stage2 cls --- (pooling select box without background)
        # training: only use the local max (hard examples)
        # testing: no difference for using all boxes or only only local max

        if self.training:
            boxes, cls_label = self.obtain_max_index(cls_s1, boxes, cls_label)
            if self.box_indices is None or not boxes.shape[0] == self.box_indices.size()[0]:
                box_indices = []
                for i in range(B):
                    box_indices.extend([i] * H)

                box_indices = box_indices * 2  # row and col are split
                self.box_indices = torch.tensor(box_indices, dtype=torch.float32)
        else:
            if self.box_indices is None or not boxes.shape[0] == self.box_indices.size()[0]:
                box_indices = []
                for i in range(B):
                    box_indices.extend([i] * H * W)
                self.box_indices = torch.tensor(box_indices, dtype=torch.float32)

            boxes = boxes.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # [K, 5]

        if jitterBox is not None:   # training
            jitter_indices = []
            for i in range(B):
                jitter_indices.extend([i] * 96)
            self.jitter_indices = torch.tensor(jitter_indices, dtype=torch.float32)
            self.jitter_indices = self.jitter_indices.to(jitterBox.device).unsqueeze(1)

            jitterBox = jitterBox.view(-1, 4)  # [K, 5]
            jitterBox = torch.cat((self.jitter_indices, jitterBox), dim=1)

            pool_fea_jitter = roi_align(fea, jitterBox, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                        sampling_ratio=-1)  # [K, C 3, 3]
            pool_fea_s3_jitter = roi_align(feas3, jitterBox, [self.roi_size, self.roi_size],
                                           spatial_scale=1. / self.stride, sampling_ratio=-1)  # [K, C 3, 3]

            pool_fea_jitter = self.down_spatial_linear(self.down_spatial_conv(pool_fea_jitter).squeeze())  # [K, C]
            pool_fea_s3_jitter = self.down_spatial_linear_s3(
                self.down_spatial_conv_s3(pool_fea_s3_jitter).squeeze())  # [K, C]

        batch_index = self.box_indices.to(boxes.device).unsqueeze(1)  # [K, 1]
        batch_box = torch.cat((batch_index, boxes), dim=1)
        pool_fea = roi_align(fea, batch_box, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                             sampling_ratio=-1)  # [K, C 3, 3]
        pool_fea_s3 = roi_align(feas3, batch_box, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                sampling_ratio=-1)  # [K, C 3, 3]

        pool_fea = self.down_spatial_linear(self.down_spatial_conv(pool_fea).squeeze())  # [K, C]
        pool_fea_s3 = self.down_spatial_linear_s3(self.down_spatial_conv_s3(pool_fea_s3).squeeze())  # [K, C]

        if self.training:
            size = pool_fea.size(0)
            pool_fea_h1 = pool_fea[:size // 2, :]
            pool_fea_h2 = pool_fea[size // 2:, :]
            pool_fea_s3_h1 = pool_fea_s3[:size // 2, :]
            pool_fea_s3_h2 = pool_fea_s3[size // 2:, :]

            pool_fea_h1 = pool_fea_h1.view(B, -1, self.inchannels // 2)
            pool_fea_h2 = pool_fea_h2.view(B, -1, self.inchannels // 2)
            pool_fea_s3_h1 = pool_fea_s3_h1.view(B, -1, self.inchannels // 2)
            pool_fea_s3_h2 = pool_fea_s3_h2.view(B, -1, self.inchannels // 2)

            target_s4 = target_s4.squeeze().unsqueeze(1)
            target_s3 = target_s3.squeeze().unsqueeze(1)

            pool_fea_h1 = pool_fea_h1 * target_s4
            pool_fea_h2 = pool_fea_h2 * target_s4
            pool_fea_s3_h1 = pool_fea_s3_h1 * target_s3
            pool_fea_s3_h2 = pool_fea_s3_h2 * target_s3

            pool_fea_h1 = pool_fea_h1.view(-1, self.inchannels // 2)
            pool_fea_h2 = pool_fea_h2.view(-1, self.inchannels // 2)
            pool_fea_s3_h1 = pool_fea_s3_h1.view(-1, self.inchannels // 2)
            pool_fea_s3_h2 = pool_fea_s3_h2.view(-1, self.inchannels // 2)

            pool_fea = torch.cat((pool_fea_h1, pool_fea_h2), dim=0)
            pool_fea_s3 = torch.cat((pool_fea_s3_h1, pool_fea_s3_h2), dim=0)

            # jitter
            pool_fea_jitter = pool_fea_jitter.view(B, -1, self.inchannels // 2)
            pool_fea_s3_jitter = pool_fea_s3_jitter.view(B, -1, self.inchannels // 2)
            pool_fea_jitter = pool_fea_jitter * target_s4
            pool_fea_s3_jitter = pool_fea_s3_jitter * target_s3
            pool_fea_jitter_merge = torch.cat((pool_fea_jitter, pool_fea_s3_jitter), dim=-1)
            pool_fea_jitter_merge = pool_fea_jitter_merge.view(-1, self.inchannels)
            pool_fea_jitter_merge = self.merge_s3s4_s2(pool_fea_jitter_merge)
            cls_jitter = self.pred_s2(pool_fea_jitter_merge)

        else:
            target_s4 = target_s4.squeeze().unsqueeze(0)
            target_s3 = target_s3.squeeze().unsqueeze(0)
            pool_fea = pool_fea * target_s4
            pool_fea_s3 = pool_fea_s3 * target_s3

        infuse_fea = torch.cat((pool_fea, pool_fea_s3), dim=1)
        infuse_fea = self.merge_s3s4_s2(infuse_fea)

        cls_s2 = self.pred_s2(infuse_fea)

        if self.training:
            outputs = {
                'cls_score_s1': cls_s1,
                'cls_score_s2': cls_s2,
                'cls_jitter': cls_jitter,
                'cls_label_s2': cls_label
            }
        else:
            outputs = {
                'cls_score_s1': cls_s1,
                'cls_score_s2': cls_s2.squeeze().view(H, W),
                'xf_conv4': fea,
                'xf_conv3': feas3,
                'zf_conv4': target_s4,
                'zf_conv3': target_s3
            }

        return outputs



class roi_template(nn.Module):
    """
    template roi pooling: get 1*1 template
    """

    def __init__(self, roi_size=3, stride=8.0, inchannels=256, alpha=0.1):
        """
        Args:
            roi_size: output size of roi
            stride: network stride
            inchannels: input channels
            alpha: for leaky-relu
        """
        super(roi_template, self).__init__()
        self.roi_size, self.stride = roi_size, float(stride)

        self.fea_encoder = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
        )

        self.fea_encoder_s3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=roi_size, stride=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
        )


        self.spatial_conv_s3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=roi_size, stride=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
        )


        self.box_indices = None

    def forward(self, boxes, fea, feas3):
        """
        Args:
            boxes: [b, 4]
            fea: [b, c, h, w]
            target_fea: [b, c]

        Returns: cls results
        """

        fea = self.fea_encoder(fea)
        feas3 = self.fea_encoder_s3(feas3)

        B, _ = boxes.size()

        if self.box_indices is None:
            box_indices = torch.arange(B, dtype=torch.float32).reshape(-1, 1)
            self.box_indices = torch.tensor(box_indices, dtype=torch.float32)

        batch_index = self.box_indices.to(boxes.device)  # [K, 1]
        batch_box = torch.cat((batch_index, boxes), dim=1)

        # ROI pooling layer
        pool_fea = roi_align(fea, batch_box, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                             sampling_ratio=-1)  # [K, C 3, 3]
        pool_fea_s3 = roi_align(feas3, batch_box, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                sampling_ratio=-1)  # [K, C 3, 3]

        # spatial resolution to 1*1
        pool_fea = self.spatial_conv(pool_fea)  # [K, C]
        pool_fea_s3 = self.spatial_conv_s3(pool_fea_s3)  # [K, C]

        if len(pool_fea.size()) == 1:
            pool_fea = pool_fea.unsqueeze(0)
            pool_fea_s3 = pool_fea_s3.unsqueeze(0)

        return pool_fea, pool_fea_s3


class SimpleMatrix(nn.Module):
    """
    Shrink feature channels (after Neck module)
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleMatrix, self).__init__()

        self.matrix11_k = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.matrix11_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, z, x):
        z11 = self.matrix11_k(z)
        x11 = self.matrix11_s(x)

        return z11, x11



class LTM(nn.Module):
    """
    Learn to match
    """
    def __init__(self, inchannel=128, used=None):
        super(LTM, self).__init__()

        if 1 in used or used is None:
            self.GuidedSP = TransGuide(inchannels=inchannel, outchannels=64)
        if 2 in used or used is None:
            self.PointDW = PointDW(inchannels=inchannel, outchannels=inchannel)
        if 3 in used or used is None:
            self.PointAdd = PointAdd(inchannels=inchannel, outchannels=inchannel)
        if 4 in used or used is None:
            self.Transformer = SimpleSelfAtt(inchannels=inchannel, outchannels=inchannel)
        if 5 in used or used is None:
            self.FiLM = FiLM(inchannels=inchannel, outchannels=inchannel)
        if 6 in used or used is None:
            self.PC = PairRelation(inchannels=inchannel, outchannels=64)
        if 7 in used or used is None:
            self.Concat = Concat(inchannels=inchannel, outchannels=inchannel)

        self.used = used
        self.branches = len(used) if used is not None else 7
        self.merge2 = nn.Parameter(1.0*torch.ones(1, self.branches * 256, 1, 1))

        # re-embedding
        # self.embed = nn.ModuleList()
        self.embed2 = nn.Sequential(
            nn.Conv2d(inchannel * self.branches, inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, xf, zf, ROI, zf_mask):

        feats = []
        used = self.used
        if 1 in used or used is None:
            feats.append(self.GuidedSP(xf, zf, zf_mask))
        if 2 in used or used is None:
            feats.append(self.PointDW(xf, ROI))
        if 3 in used or used is None:
            feats.append(self.PointAdd(xf, ROI))
        if 4 in used or used is None:
            feats.append(self.Transformer(xf, zf))
        if 5 in used or used is None:
            feats.append(self.FiLM(xf, ROI))
        if 6 in used or used is None:
            feats.append(self.PC(xf, zf))
        if 7 in used or used is None:
            feats.append(self.Concat(xf, ROI))

        feats2 = torch.cat(feats, dim=1)
        merge = F.sigmoid(self.merge2)
        feats3 = merge * feats2
        out = self.embed2(feats3)

        return out


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor(tensor1: torch.Tensor):
    # TODO make this more general
    tensor1 = tensor1.squeeze(0)
    if tensor1[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor1])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor1)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor1.dtype
        device = tensor1.device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor1, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def nested_tensor_from_tensor_2(tensor1: torch.Tensor):
    # TODO make this more general
    if tensor1[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor1])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor1)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor1.dtype
        device = tensor1.device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor1, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

