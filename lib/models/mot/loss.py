''' Details
Author: Zhipeng Zhang/Chao Liang
Function: train MOT methods
Date: 2022.4.7
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.general_helper import is_parallel
from utils.box_helper import bbox_iou


class JDELoss(torch.nn.Module):
    def __init__(self, nID, emb_dim):
        """
        build JDE series training loss
        :param nID: total target id number in training set
        :param emb_dim: embedding feature channels
        """
        super(JDELoss, self).__init__()

        self.nID = nID
        self.emb_dim = emb_dim

        self.bottleneck = nn.BatchNorm1d(self.emb_dim).cuda()
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.emb_dim, self.nID, bias=False).cuda()
        self.IDLoss_zero = nn.CrossEntropyLoss(ignore_index=0).cuda()
        self.IDLoss = CrossEntropyLabelSmooth(self.nID).cuda()
        if self.nID == 1: self.nID += 1  # binary classifictaion
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

    def forward(self, output, targets, model):
        id_embeding, p = output[0], output[1][0]  # id embedding and box
        device = targets.device
        lcls, lbox, lobj, id_loss, lrep, lrep0 = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device),\
                                                 torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, indices_id, tids = build_targets(p, targets, model)  # targets
        h = model.hyp  # hyperparameters

        self.s_id = nn.Parameter(0.02*torch.ones(1)).to(device)

        # define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h.TRAIN.CLS_P_WEIGHT])).to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h.TRAIN.OBJ_P_WEIGHT])).to(device)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.2)

        # focal loss
        g = h.TRAIN.FL_GAMMA  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # losses
        nt = 0  # number of targets
        np = len(p)  # number of outputs
        balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).mean()  # giou loss

                # objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

                # classification
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:6], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:6], t)  # BCE

                # id_loss
                if i == 0:
                    ps_id = id_embeding[indices_id[i]]
                    id_head = self.emb_scale * F.normalize(ps_id).to(device)
                    id_output = self.classifier(id_head).contiguous()
                    if len(tids[i]) > 1 and len(id_output) > 1:
                        id_target = tids[i].squeeze()
                        id_loss_zero = self.IDLoss_zero(id_output, id_target)
                        '''
                        index = 0
                        while 1:
                            if index == len(id_target):
                                break
                            if id_target[index] == 0:
                                id_target = id_target[torch.arange(id_target.size(0)) != index]
                                id_output = id_output[torch.arange(id_output.size(0)) != index]
                            else:
                                index += 1
                        if len(id_target) > 0:
                            id_loss += self.IDLoss(id_output, id_target)
                        else:
                            id_loss += id_loss_zero
                        '''
                        id_loss += id_loss_zero

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        s = 3 / np  # output count scaling
        lbox *= h.TRAIN.GIOU_WEIGHT * s
        lobj *= h.TRAIN.OBJ_WEIGHT * s * (1.4 if np == 4 else 1.)
        lcls *= h.TRAIN.CLS_WEIGHT * s
        bs = tobj.shape[0]  # batch size

        # 0.02 for s_id is good for only mot17
        loss = lbox + lobj + id_loss*self.s_id
        return loss * bs, torch.cat((lbox, id_loss, lobj, loss)).detach()


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(),
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def build_targets(p, targets, model):
    """
    parser model output for loss computation
    :param p:
    :param targets: (image,class,x,y,w,h)
    :param model:
    :return:
    """

    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    indices_id, tids = [], []
    gain = torch.ones(7, device=targets.device)  # normalized to grid-space gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # [na, target_num, 7] append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain

        # Define
        b, c = t[0][:, :2].long().T  # image, class
        gxy = t[0][:, 2:4]  # grid xy
        gwh = t[0][:, 4:6]  # grid wh
        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices
        # Append
        indices_id.append((b, gj, gi))  # image, anchor, grid indices
        tids.append(c)  # class

        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp.TRAIN.ANCHOR_THRESH  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch, indices_id, tids


def smooth_BCE(eps=0.1):
    """
    return positive, negative label smoothing BCE targets
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    :param eps:
    :return:
    """

    return 1.0 - 0.5 * eps, 0.5 * eps


def labels_to_class_weights(labels, nc=80):
    """
    Get class weights (inverse frequency) from training labels
    used for JDE series
    :param labels:
    :param nc:
    :return:
    """

    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """
    get weights for each image in a batch
    :param labels:
    :param nc:
    :param class_weights:
    :return:
    """
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights