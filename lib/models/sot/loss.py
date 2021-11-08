''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: loss functions for SOT
Data: 2021.6.23
'''

import torch
import torch.nn as nn

BCE_TORCH = nn.BCEWithLogitsLoss()

# modules
def simple_BCE(pred, label, select):
    '''
    binary cross-entropy for selected elements (predictions)
    '''
    if len(select.size()) == 0: return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return BCE_TORCH(pred, label)  # the same as tf version


def weighted_BCE(pred, label):
    '''
    weighted binary cross-entropy: 0.5*pos + 0.5*neg
    '''
    pred = pred.view(-1)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()

    loss_pos = simple_BCE(pred, label, pos)
    loss_neg = simple_BCE(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


# used in other scripts
def WBCE(pred, label):
    '''
    weighted binary cross-entropy: 0.5*pos + 0.5*neg
    used in SiamFC, SiamDW, Ocean, AutoMatch
    '''
    loss = weighted_BCE(pred, label)
    return loss


def WBCE_ADDPOS(pred, label, jitter, jitter_label):
    '''
    add more (hard) positive examples to balance negative ones
    weighted binary cross-entropy: 0.5*(pos_ori + pos_add) + 0.5*neg
    used in AutoMatch
    '''
    pred = pred.view(-1)
    label = label.view(-1)

    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_neg = simple_BCE(pred, label, neg)

    if len(pos) > 0:
        loss_pos = simple_BCE(pred, label, pos)
        jitter_loss = BCE_TORCH(jitter.view(-1), jitter_label.view(-1))
        return (loss_pos + jitter_loss) * 0.5 + loss_neg * 0.5
    else:
        return loss_neg


def WBCEwithAILGN(pred, label, align_pred, align_label):
    '''
    WBCE for both original and object-aware classification
    used in Ocean
    '''
    loss_ori = weighted_BCE(pred, label)
    loss_align = weighted_BCE(align_pred, align_label)
    return loss_ori + loss_align


def IOULoss(pred, target, weight=None):
    '''
    IOU loss used in FCOS format
    '''
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum() / weight.sum()
    else:
        assert losses.numel() != 0
        return losses.mean()


def FCOSIOU(bbox_pred, reg_target, reg_weight):
    '''
    FCOS output format IOU loss for regression.
    '''

    bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    reg_target_flatten = reg_target.reshape(-1, 4)
    reg_weight_flatten = reg_weight.reshape(-1)
    pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

    bbox_pred_flatten = bbox_pred_flatten[pos_inds]
    reg_target_flatten = reg_target_flatten[pos_inds]

    loss = IOULoss(bbox_pred_flatten, reg_target_flatten)

    return loss



