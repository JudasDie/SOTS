''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: data augmenttaion
Data: 2021.6.23
'''

import utils.box_helper as boxhelper


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        center = boxhelper.corner2center(bbox)
        original_center = center

        real_param = {}
        if 'scale' in param:
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)
            center = boxhelper.Center(center.x, center.y, center.w * scale_x, center.h * scale_y)

        bbox = boxhelper.center2corner(center)

        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = boxhelper.Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = boxhelper.Corner(*map(round, bbox))

        current_center = boxhelper.corner2center(bbox)

        real_param['scale'] = current_center.w / original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - original_center.x, current_center.y - original_center.y

        return bbox, real_param
    else:
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.

        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0

        center = boxhelper.corner2center(bbox)

        center = boxhelper.Center(center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y)

        return boxhelper.center2corner(center)
