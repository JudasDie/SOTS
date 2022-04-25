''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: NMS for detection
Data: 2022.4.7
'''

import torch


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep.type(torch.long)


def cluster_nms(boxes, scores, iou_threshold: float = 0.5, top_k: int = 200):
    # Collapse all the classes into 1
    _, idx = scores.sort(0, descending=True)
    # idx = idx[:top_k]
    boxes_idx = boxes[idx]
    iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    idx_out = idx[maxA <= iou_threshold]
    return idx_out


def cluster_diounms(boxes, scores, iou_threshold: float = 0.5, dense_mask=[], top_k: int = 200):
    # Collapse all the classes into 1
    _, idx = scores.sort(0, descending=True)
    # idx = idx[:top_k]
    boxes_idx = boxes[idx]
    iou = diou(boxes_idx, boxes_idx, delta=0.7).triu_(diagonal=1)
    B = iou
    if len(dense_mask) != 0:
        x_inds = (boxes_idx[:, 0] + boxes_idx[:, 2]) // 16
        y_inds = (boxes_idx[:, 1] + boxes_idx[:, 3]) // 16
        y_inds[y_inds >= 76] = 75
        y_inds[y_inds < 0] = 0
        x_inds[x_inds >= 136] = 135
        x_inds[x_inds < 0] = 0
        x_inds = x_inds.cpu().numpy().astype(np.int16).tolist()
        y_inds = y_inds.cpu().numpy().astype(np.int16).tolist()
        dense_mask = dense_mask.squeeze(dim=0).squeeze(dim=0)
        dense_mask = dense_mask[y_inds, x_inds].cuda()
        dense_mask[dense_mask <= iou_threshold] = iou_threshold
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        if len(dense_mask) != 0:
            E = (torch.lt(maxA, dense_mask)).float().unsqueeze(1).expand_as(A)
        else:
            E = (torch.lt(maxA, iou_threshold)).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    if len(dense_mask) != 0:
        idx_out = idx[torch.lt(maxA, dense_mask)]
    else:
        idx_out = idx[maxA <= iou_threshold]
    return idx_out


def cluster_SPM_nms(boxes, scores, iou_threshold: float = 0.5, top_k: int = 200):
    # Collapse all the classes into 1
    _, idx = scores.sort(0, descending=True)
    boxes_idx = boxes[idx]
    scores = scores[idx]
    boxes = boxes_idx
    iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    scores = torch.prod(torch.exp(-B ** 2 / 0.2), 0) * scores
    idx_out = scores > 0.01
    return idx[idx_out]


def cluster_SPM_dist_nms(boxes, scores, iou_threshold: float = 0.5, top_k: int = 200):
    # Collapse all the classes into 1
    _, idx = scores.sort(0, descending=True)
    boxes_idx = boxes[idx]
    scores = scores[idx]
    boxes = boxes_idx
    iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    D = distance(boxes, boxes, delta=0.7)
    X = (B >= 0).float()
    scores = torch.prod(torch.min(torch.exp(-B ** 2 / 0.2) + D * ((B > 0).float()), X), 0) * scores
    idx_out = scores > 0.15

    return idx[idx_out]


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b, iscrowd=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    if iscrowd:
        return inter / area_a
    else:
        return inter / union  # [A,B]


def diou(box_a, box_b, delta=0.9, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
        inter = inter[None, ...]

    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)
    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7))
    out = inter / area_a if iscrowd else inter / union - D ** delta
    return out if use_batch else out.squeeze(0)


def d2iou(box_a, box_b, delta=0.9, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
        inter = inter[None, ...]

    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)
    w1 = ((box_a[:, :, 2] - box_a[:, :, 0])).unsqueeze(2).expand_as(inter)
    h1 = ((box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)
    w2 = ((box_b[:, :, 2] - box_b[:, :, 0])).unsqueeze(1).expand_as(inter)
    h2 = ((box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)
    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    # D = (((x2 - x1)**2 + (y2 - y1)**2) / ((cr-cl)**2 + (cb-ct)**2 + 1e-7))
    # Deform = (torch.abs(torch.log(w1/w2))+torch.abs(torch.log(h1/h2))+torch.abs(torch.log((w1*h1)/(w2*h2))))**2.5
    D = torch.max(((x2 - x1) ** 2) / ((cr - cl) ** 2 + 1e-7), ((y2 - y1) ** 2) / ((cb - ct) ** 2 + 1e-7))
    out = inter / area_a if iscrowd else inter / union - D ** delta
    return out if use_batch else out.squeeze(0)


def distance(box_a, box_b, delta=0.9, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
        inter = inter[None, ...]

    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)

    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7)) ** delta
    out = D if iscrowd else D
    return out if use_batch else out.squeeze(0)


def speed():
    boxes = 1000 * torch.rand((1000, 100, 4), dtype=torch.float)
    boxscores = torch.rand((1000, 100), dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    start = time.time()
    for i in range(1000):
        soft_nms_pytorch(boxes[i], boxscores[i], cuda=cuda)
    end = time.time()
    print("Average run time: %f ms" % (end - start))


def test():
    # boxes and boxscores
    boxes = torch.tensor([[200, 200, 400, 400],
                          [220, 220, 420, 420],
                          [200, 240, 400, 440],
                          [240, 200, 440, 400],
                          [1, 1, 2, 2]], dtype=torch.float)
    boxscores = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9], dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    print(soft_nms_pytorch(boxes, boxscores, cuda=cuda))


if __name__ == '__main__':
    test()