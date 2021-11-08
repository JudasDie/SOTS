''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: supporting functions during tracking phase
Data: 2021.6.23
'''
import cv2
import math
import torch
import numpy as np
from .box_helper import *

def siam_crop(crop_input, mode='torch'):
    """
    cropping image for tracking in Siamese framework
    """
    im, pos, model_sz, original_sz, avg_chans = crop_input['image'], crop_input['pos'], crop_input['model_sz'], \
                                                crop_input['original_sz'], crop_input['avg_chans']

    if len(im.shape) == 2:
        mask_format = True
    else:
        mask_format = False

    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    if not mask_format:
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
            # for return mask
            tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1),:]
        else:
            tete_im = np.zeros(im.shape[0:2])
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
    else:
        r, c = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad), np.uint8)
            # for return mask
            tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

            te_im[top_pad:top_pad + r, left_pad:left_pad + c] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c] = 0
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c] = 0
            if left_pad:
                te_im[:, 0:left_pad] = 0
            if right_pad:
                te_im[:, c + left_pad:] = 0
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
        else:
            tete_im = np.zeros(im.shape[0:2])
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if mode == 'torch':
        crop_output = {'image_tensor': im_to_torch(im_patch.copy()), 'meta_info': crop_info}
    else:
        crop_output = {'image_tensor': im_patch, 'meta_info': crop_info}
    return crop_output


def siamfc_pyramid_crop(inputs):
    """
    crop siamfc pyramid images
    """
    im, pos, in_side_scaled, out_side, avg_chans = inputs['image'], inputs['pos'], inputs['scaled_instance'], \
                                                   inputs['instance_size'], inputs['avg_chans']

    in_side_scaled = [round(x) for x in in_side_scaled]
    num_scale = len(in_side_scaled)
    pyramid = torch.zeros(num_scale, 3, out_side, out_side)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side

    search_side = round(beta * max_target_side)
    crop_input = {'image': im, 'pos': pos, 'model_sz': int(search_side),
                  'original_sz': int(max_target_side),
                  'avg_chans': avg_chans}

    out1 = siam_crop(crop_input, mode='numpy')
    search_region = out1['image_tensor']

    for s, temp in enumerate(in_side_scaled):
        target_side = round(beta * temp)
        crop_input = {'image': search_region, 'pos': (1 + search_side) / 2, 'model_sz': out_side,
                      'original_sz': target_side,
                      'avg_chans': avg_chans}


        temp = siam_crop(crop_input)
        pyramid[s, :] = temp['image_tensor']

    crop_output = {'image_tensor': pyramid, 'meta_info': None}
    return crop_output


def im_to_torch(img):
    """
    numpy image to pytorch tensor
    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.from_numpy(img).float()
    return img

def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)


def generate_psedou_mask(target_pos, target_sz, img_sz):
    """
    generate psedou mask for OceanPlus and AutoMatch
    where we set the pixel in bbox as 1, outside that as 0
    """
    img_h, img_w = img_sz
    cx, cy = target_pos
    w, h = target_sz
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1, y1, x2, y2 = int(x1), int(y1), math.ceil(x2 + 1), math.ceil(y2 + 1)
    mask = np.zeros((img_sz[0], img_sz[1]))  # h,w
    mask[y1:y2, x1:x2] = 1

    return mask


def get_bbox(s_z, p, tsz):
    """
    map the GT bounding box in the first frame to template (127*127)
    """
    exemplar_size = p.exemplar_size
    scale_z = exemplar_size / s_z
    w, h = tsz[0], tsz[1]
    imh, imw = p.exemplar_size, p.exemplar_size
    w = w*scale_z
    h = h*scale_z
    cx, cy = imw//2, imh//2
    bbox = center2corner([cx, cy, w, h])
    return bbox   # [x1, y1, x2, y2]

def jitter_shift():
    """
    jitter box (for zoom trick in AutoMatch)
    """
    add = np.array([4, 8, 12, 16]).astype(np.float)
    minus = -1 * add
    add2 = add.reshape(4, 1).repeat(2, axis=-1)
    minus2 = minus.reshape(4, 1).repeat(2, axis=1)

    if True:
        shift = np.zeros((96, 4))

        # settle (x1, y1) change (x2, y2)
        shift[0:4, 2] += add
        shift[4:8, 2] += minus
        shift[8:12, 3] += add
        shift[12:16, 3] += minus
        shift[16:20, 2:4] += add2
        shift[20:24, 2:4] += minus2

        # settle (x2, y1) change (x1, y2)
        shift[24:28, 0] += add
        shift[28:32, 0] += minus
        shift[32:36, 3] += add
        shift[36:40, 3] += minus
        shift[40:44, 0] += add
        shift[40:44, 3] += add
        shift[44:48, 0] += minus
        shift[44:48, 3] += minus

        # settle (x2, y2) change (x1, y1)
        shift[48:52, 0] += add
        shift[52:56, 0] += minus
        shift[56:60, 1] += add
        shift[60:64, 1] += minus
        shift[64:68, 0:2] += add2
        shift[68:72, 0:2] += minus2

        # settle (x1, y2) change (x2, y1)
        shift[72:76, 2] += add
        shift[76:80, 2] += minus
        shift[80:84, 1] += add
        shift[84:88, 1] += minus
        shift[88:92, 1:3] += add2
        shift[92:96, 1:3] += minus2

    return shift
