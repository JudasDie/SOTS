''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: loss functions for SOT
Data: 2021.6.23
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    pos = label.data.eq(1).nonzero(as_tuple=False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple=False).squeeze().cuda()

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


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

def AllIOULoss(box1, box2, weight=None, xyxy=False, xywh=False, ltrb=False, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box1 = box1.t()
    # box2 = box2.t()
    # Get the coordinates of bounding boxes
    if ltrb:
        b1_x1 = 128.0 - box1[:, 0]
        b1_x2 = 128.0 + box1[:, 2]
        b1_y1 = 128.0 - box1[:, 1]
        b1_y2 = 128.0 + box1[:, 3]
        b2_x1 = 128.0 - box2[:, 0]
        b2_x2 = 128.0 + box2[:, 2]
        b2_y1 = 128.0 - box2[:, 1]
        b2_y2 = 128.0 + box2[:, 3]
    elif xywh:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2
        b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2
        b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
        b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2
    elif xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]


    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            losses = iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                losses = iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                losses = iou - (rho2 / c2 + v * alpha)  # CIoU
    else:
        losses = iou
    # losses = -torch.log((1.-losses)/2.)
    losses = 1. - losses
    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum() / weight.sum()
    else:
        assert losses.numel() != 0
        return losses

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def TransInMoRegLoss(bbox_pred, reg_target, indices):
    num_boxes = bbox_pred.size(1)
    idx = _get_src_permutation_idx(indices)
    bbox_pred = bbox_pred[idx]
    reg_target = reg_target.unsqueeze(1)
    reg_target = torch.cat([t[i] for t, (_, i) in zip(reg_target, indices)], dim=0)

    bbox_pred = box_cxcywh_to_xyxy(bbox_pred)
    loss_bbox = F.l1_loss(bbox_pred, reg_target, reduction='none')

    losses = {}
    losses['loss_l1'] = loss_bbox.sum() / num_boxes

    losses['loss_iou'] = AllIOULoss(bbox_pred, reg_target, xyxy=True, CIoU=True).sum() / num_boxes
    return losses


def TransInMoClsLoss(cls_pred, cls_label, indices, cfg):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    src_logits = cls_pred

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(cls_label, indices)]).long()
    target_classes = torch.full(src_logits.shape[:2], 1,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    empty_weight = torch.ones(2)
    empty_weight[-1] = cfg.MODEL.LOSS.EOS_WEIGHT
    loss = F.cross_entropy(src_logits.reshape(-1, 2), target_classes.view(-1), empty_weight.to(src_logits.device))
    return loss

INF = 100000000


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero(as_tuple=False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple=False).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

class IOULoss_module(nn.Module):
    def iou_loss(self, pred, target, weight=None):
        loss_bbox = F.l1_loss(pred, target, reduction='elementwise_mean')

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def forward(self, box1, box2, weight=None, xyxy=False, xywh=False, ltrb=True, GIoU=False, DIoU=False, CIoU=True):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        # box1 = box1.t()
        # box2 = box2.t()
        # loss_bbox = F.l1_loss(box1, box2, reduction='none').sum(dim=-1)

        # Get the coordinates of bounding boxes
        if ltrb:
            b1_x1 = 128.0 - box1[:, 0]
            b1_x2 = 128.0 + box1[:, 2]
            b1_y1 = 128.0 - box1[:, 1]
            b1_y2 = 128.0 + box1[:, 3]
            b2_x1 = 128.0 - box2[:, 0]
            b2_x2 = 128.0 + box2[:, 2]
            b2_y1 = 128.0 - box2[:, 1]
            b2_y2 = 128.0 + box2[:, 3]
        elif xywh:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        elif xyxy:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = (w1 * h1 + 1e-16) + w2 * h2 - inter

        iou = inter / union  # iou
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + 1e-16  # convex area
                losses = iou - (c_area - union) / c_area  # GIoU
            if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                # convex diagonal squared
                c2 = cw ** 2 + ch ** 2 + 1e-16
                # centerpoint distance squared
                rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
                if DIoU:
                    losses = iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (1 - iou + v)
                    losses = iou - (rho2 / c2 + v * alpha)  # CIoU
        else:
            losses = iou
        # losses = -torch.log((1.-losses)/2.)
        losses = 1. - losses

        weight_sum = weight.sum()
        losses = (losses * weight).sum() / weight_sum
        return losses

def compute_locations(features,stride):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + 32  #alex:48 // 32 (256-w*8)/2+4
    return locations


class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self, output_size=25):
        self.box_reg_loss_func = IOULoss_module()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.output_size = output_size

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        bboxes = gt_bbox
        labels = labels.view(self.output_size**2,-1)


        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        # print(bboxes[:,2]-bboxes[:,0])
        # print((bboxes[:,3]-bboxes[:,1]))

        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1

        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, preds, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # labels = labels.transpose(0,1).reshape(labels.size(0), self.output_size, self.output_size)
        box_cls, box_regression, centerness = preds['cls'], preds['reg'], preds['cen']
        locations = compute_locations(box_cls, 8)
        box_cls = self.log_softmax(box_cls)

        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten.long())

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss
