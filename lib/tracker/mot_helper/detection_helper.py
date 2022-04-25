''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: Detection in MOT
Data: 2022.4.7
'''

import time
import torch
import torchvision
import numpy as np
import utils.box_helper as box_helper
import tracker.mot_helper.nms_helper as nms_helper


def non_max_suppression_and_inds(prediction, conf_thres=0.1, iou_thres=0.6, dense_mask=[], merge=False,
                                 classes=None, agnostic=False, method='standard'):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = 1  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:6] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = box_helper.xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:6] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:6].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 6:]), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        if method == 'standard':
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if method == 'soft':
            i = nms_helper.soft_nms_pytorch(boxes, scores, sigma=0.5, thresh=0.2, cuda=1)
        if method == "cluster":
            i = nms_helper.cluster_nms(boxes, scores, iou_thres)
        if method == "cluster_SPM":
            i = nms_helper.cluster_SPM_nms(boxes, scores, iou_thres)
        if method == "cluster_diou":
            i = nms_helper.cluster_diounms(boxes, scores, iou_thres, dense_mask)
        if method == "cluster_SPM_dist":
            i = nms_helper.cluster_SPM_dist_nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_helper.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    if len(x) != 0:
        x_inds = (output[0][:, 0] + output[0][:, 2]) // 16
        y_inds = (output[0][:, 1] + output[0][:, 3]) // 16
        y_inds[y_inds >= 76] = 75
        # y_inds[y_inds < 0] = 0
        x_inds[x_inds >= 136] = 135
        # x_inds[x_inds < 0] = 0
        x_inds = x_inds.cpu().numpy().tolist()
        y_inds = y_inds.cpu().numpy().tolist()
        # x_inds = [int(x) for x in x_inds]
        # y_inds = [int(x) for x in y_inds]
    else:
        return [], [], []
    return output[0].cpu(), x_inds, y_inds


def non_max_suppression_jde(prediction, conf_thres=0.5, nms_thres=0.4, method='standard'):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = box_helper.xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        if method == 'standard':
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == 'fast':
            nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres)
        else:
            raise ValueError('Invalid NMS type!')
        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    if len(output[0]) != 0:
        x_inds = (output[0][:, 0] + output[0][:, 2]) // 16
        y_inds = (output[0][:, 1] + output[0][:, 3]) // 16
        x_inds = x_inds.cpu().numpy().tolist()
        y_inds = y_inds.cpu().numpy().tolist()
        # x_inds = [int(x) for x in x_inds]
        # y_inds = [int(x) for x in y_inds]
    else:
        return [], [], []
    return output[0].cpu(), x_inds, y_inds


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = 1  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:6] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = box_helper.xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:6] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:6].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 6:]), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_helper.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output



def scale_coords(img_size, coords, img0_shape):
    """
    Rescale x1, y1, x2, y2 from 416 to image size
    :param img_size: (width, height)
    :param coords: x1y1x2y2
    :param img0_shape: (width, height)
    :return:
    """

    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    # coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def clip_coords(boxes, img_shape):
    """
    clip bounding xyxy bounding boxes to image shape (height, width)
    :param boxes: xyxy
    :param img_shape: (height, width)
    :return:
    """
    r = 0.1
    boxes[:, 0].clamp_(-r*img_shape[1], img_shape[1]+r*img_shape[1])  # x1
    boxes[:, 1].clamp_(-r*img_shape[0], img_shape[0]+r*img_shape[0])  # y1
    boxes[:, 2].clamp_(-r*img_shape[1], img_shape[1]+r*img_shape[1])  # x2
    boxes[:, 3].clamp_(-r*img_shape[0], img_shape[0]+r*img_shape[0])  # y2


def output_to_target(output, width, height):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                if isinstance(pred, torch.Tensor) and pred.is_cuda:
                    pred = pred.cpu()
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)