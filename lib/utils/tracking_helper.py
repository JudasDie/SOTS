''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: supporting functions during tracking phase
Data: 2021.6.23
'''
import os
import os.path as osp
import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
from .box_helper import *
from .log_helper import logger
import pdb


# ----------------------------- MOT -------------------------------
def parser_mot_train_data(cfg):
    """
    parser training and validation data
    :param cfg:
    :return:
    """
    mode = cfg.TRAIN.DATASET.WHICH_MODE
    train_use, val_use = cfg.TRAIN.DATASET.CONFIG[mode].TRAIN_USE, cfg.TRAIN.DATASET.CONFIG[mode].VAL_USE
    train_set, val_set = dict(), dict()

    cur_path = osp.dirname(__file__)
    for data in train_use:
        train_set[data] = osp.join(cur_path, '../dataset/mot_imgs/{}'.format(data.replace('_', '.')))

    for data in val_use:
        val_set[data] = osp.join(cur_path, '../dataset/mot_imgs/{}'.format(data.replace('_', '.')))

    return train_set, val_set


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
     Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scaleFill:
    :param scaleup:
    :return:
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def letterbox_jde(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    """
    resize and pad a image to network input size
    :param img:
    :param height: height for network input
    :param width: width for network input
    :param color:
    :return:
    """

    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def update_cstrack_hypers(opt, args, config):
    """
    update hyper-parameters of mot models
    :param opt: edict, output
    :param args: script arg-parser input
    :param config: .yaml file configures (experiments/xx.yaml)
    :return: opt
    """

    opt.args = args
    opt.cfg = config

    # copy all keys and values of cfg to opt
    for k, v in config.items():
        opt[k] = v

    cfg_hypers = config.TEST.COMMON_HYPER if args.benchmark is None else config.TEST[args.benchmark]  # special benchmarks in .yaml file

    opt.nms_thres = args.nms_thres if args.nms_thres is not None else cfg_hypers.nms_thres
    opt.conf_thres = args.conf_thres if args.conf_thres is not None else cfg_hypers.conf_thres
    opt.track_buffer = args.track_buffer if args.track_buffer is not None else cfg_hypers.track_buffer
    opt.min_box_area = args.min_box_area if args.min_box_area is not None else cfg_hypers.min_box_area
    opt.img_size = args.img_size if args.img_size is not None else tuple(cfg_hypers.img_size)
    opt.mean = args.mean if args.mean is not None else cfg_hypers.mean
    opt.std = args.std if args.std is not None else cfg_hypers.std
    
    return opt


def get_mot_benchmark_path(opt):
    curr_path = osp.realpath(osp.dirname(__file__))

    if opt.args.val_mot15:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot15_train.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT15/train')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT15/images/train')
        benchmark_name = 'MOT15'

    elif opt.args.test_mot15:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot15_test.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT15/test')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT15/images/test')
        benchmark_name = 'MOT15'

    elif opt.args.val_mot16:  # training sequences
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot16_train.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT16/train')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT16/images/train')
        benchmark_name = 'MOT16'

    elif opt.args.test_mot16:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot16_test.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT16/test')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT16/images/test')
        benchmark_name = 'MOT16'

    elif opt.args.val_mot17:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot17_train.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT17/train')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT17/images/train')
        benchmark_name = 'MOT17'

    elif opt.args.test_mot17:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot17_test.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT17/test')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT17/images/test')
        benchmark_name = 'MOT17'

    elif opt.args.val_mot20:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot20_train.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT20/train')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT20/images/train')
        benchmark_name = 'MOT20'

    elif opt.args.test_mot20:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', 'mot20_test.txt')).readlines()
        data_root = osp.join(opt.args.data_dir, 'MOT20/test')
        if not osp.isdir(data_root): data_root = os.path.join(opt.args.data_dir, 'MOT20/images/test')
        benchmark_name = 'MOT20'

    else:
        seqs = open(osp.join(curr_path, '../dataset/mot_videos', '{}.txt').format(opt.args.benchmark)).readlines()
        data_root = osp.join(opt.args.data_dir, opt.args.benchmark)
        benchmark_name = opt.args.benchmark

    logger.info('testing videos: '.format(seqs))
    logger.info('data path: '.format(data_root))

    return seqs, data_root, benchmark_name

# ----------------------------- SOT -------------------------------
def siam_crop(crop_input, mode='torch', pysot_crop=False):
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
    if pysot_crop:
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_ymin = np.floor(pos[1] - c + 0.5)
    else:
        context_xmin = round(pos[0] - c)
        context_ymin = round(pos[1] - c)
    context_xmax = context_xmin + sz - 1
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

def bbox_clip(x, min_value, max_value):
    new_x = max(min_value, min(x, max_value))
    return new_x
