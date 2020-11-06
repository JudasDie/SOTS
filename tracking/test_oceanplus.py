import _init_paths
import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
import models.models as models
from tqdm import tqdm
try:
    from torch2trt import TRTModule
except:
    print('Warning: TensorRT is not successfully imported')
from PIL import Image
from os.path import exists, join, dirname, realpath
from tracker.oceanplus import OceanPlus
from tracker.online import ONLINE
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
from eval_toolkit.pysot.datasets import VOTDataset
from eval_toolkit.pysot.evaluation import EAOBenchmark
from core.eval_got10k import eval_got10k_tune
import pdb

def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', dest='arch', default='OceanPlus', choices=['OceanPlus', 'OceanPlusTRT'], help='backbone architecture')
    parser.add_argument('--mms', default='True', type=str, choices=['True', 'False'], help='wether to use MMS')
    parser.add_argument('--resume', default="snapshot/OceanPlusMSS.pth", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--online', action="store_true", help='whether to use online')
    parser.add_argument('--vis', action="store_true", help='visualize tracking results')
    parser.add_argument('--hp', default=None, type=str, help='hyper-parameters')
    parser.add_argument('--debug', default=False, type=str, help='debug or not')
    args = parser.parse_args()

    return args


def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0

    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)
    mask = rle_to_mask(rle, region_w, region_h)

    return mask

def save_prediction(prediction, palette, save_path, save_name):
    if prediction.ndim > 2:
        img = Image.fromarray(np.uint8(prediction[0, ...]))
    else:
        img = Image.fromarray(np.uint8(prediction))
    img = img.convert('L')
    img.putpalette(palette)
    img = img.convert('P')
    img.save('{}/{}.png'.format(save_path, save_name))


def track(siam_tracker, online_tracker, siam_net, video, args):
    """
    track a single video in VOT2020
    attention: not for benchmark evaluation, just a demo
    TODO: add cyclic initiation
    """

    start_frame, toc = 0, 0
    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if args.online:
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            lx, ly, w, h = eval(gt[f][1:])[:4]
            cx = lx + w / 2
            cy = ly + h / 2

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            mask_roi = create_mask_from_string(eval(gt[f][1:]))
            hi, wi, _ = im.shape
            mask_gt = np.zeros((hi, wi))
            mask_gt[ly:ly + h, lx:lx + w] = mask_roi

            state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask_gt, debug=args.debug)  # init siamese tracker

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

        elif f > start_frame:  # tracking
            if args.online:
                state = online_tracker.track(im, rgb_im, siam_tracker, state)
            else:
                state = siam_tracker.track(state, im, name=image_file)
            mask = state['mask']

            if args.vis:
                COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
                mask = COLORS[mask]
                output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
                cv2.imshow("mask", output)
                cv2.waitKey(1)

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def track_vos(siam_tracker, online_tracker, siam_net, video, args, hp=None):
    re = args.resume.split('/')[-1].split('.')[0]

    if hp is None:
        save_path = join('result', args.dataset, re, video['name'])
    else:
        # re = re+'_thr_{:.2f}_lambdaU_{:.2f}_lambdaS_{:.2f}_iter1_{:.2f}_iter2_{:.2f}'.format(hp['seg_thr'], hp['lambda_u'], hp['lambda_s'], hp['iter1'], hp['iter2'])
        re = re+'_pk_{:.3f}_wi_{:.2f}_lr_{:.2f}'.format(hp['penalty_k'], hp['window_influence'], hp['lr'])
        save_path = join('result', args.dataset, re, video['name'])

    if exists(save_path):
        return

    image_files = video['image_files']
    annos = [Image.open(x) for x in video['anno_files'] if exists(x)]
    palette = annos[0].getpalette()
    annos = [np.array(an) for an in annos]

    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    mot_enable = args.dataset in ['DAVIS2017', 'YTBVOS']

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    object_num = len(object_ids)
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))
    for obj_id, o_id in enumerate(object_ids):
        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            if args.online:
                rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            tic = cv2.getTickCount()
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id
                mask = mask.astype(np.uint8)
                x, y, w, h = cv2.boundingRect(mask)
                cx, cy = x + w/2, y + h/2
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask, hp=hp, debug=args.debug)  # init tracker

                if args.online:
                    online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset,
                                        resume=args.resume)
                pred_masks[obj_id, f, :, :] = mask
            elif end_frame >= f > start_frame:  # tracking
                if args.online:
                    state = online_tracker.track(im, rgb_im, siam_tracker, state, name=image_file)
                else:
                    state = siam_tracker.track(state, im, name=image_file)
                mask = state['mask']   # binary
                mask_ori = state['mask_ori']   # probabilistic
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                if f == start_frame:
                    pred_masks[obj_id, f, :, :] = mask
                else:
                    if args.dataset in ['DAVIS2017', 'YTBVOS']:   # multi-object
                        pred_masks[obj_id, f, :, :] = mask_ori
                    else:
                        pred_masks[obj_id, f, :, :] = mask

            if args.vis:
                COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
                mask = COLORS[mask]
                output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
                cv2.imshow("mask", output)
                cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save for evaluation

    if not exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'DAVIS2016':
        for idx in range(f+1):
            save_name = str(idx).zfill(5)
            save_prediction(pred_masks[:, idx, ...], palette, save_path, save_name)
    elif args.dataset in ['DAVIS2017', 'YTBVOS']:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for idx in range(f+1):
            if not args.dataset == 'YTBVOS':
                save_name = str(idx).zfill(5)
            else:
                save_name = image_files[idx].split('/')[-1].split('.')[0]

            save_prediction(pred_mask_final[idx, ...], palette, save_path, save_name)
    else:
        raise ValueError('not supported dataset')

    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, (f+1) / toc))

def track_box(siam_tracker, online_tracker, siam_net, video, args):
    """
    track a benchmark with only box annoated
    attention: not for benchmark evaluation, just a demo
    """

    tracker_path = os.path.join('result', args.dataset, args.arch)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = []
    b_overlaps, b_overlaps2, b_overlaps3 = [], [], []
    lost = 0
    start_frame, toc = 0, 0
    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if args.online:
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            mask_gt = None

            state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask_gt, debug=args.debug)  # init siamese tracker

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

        elif f > start_frame:  # tracking
            if args.online:
                state = online_tracker.track(im, rgb_im, siam_tracker, state)
            else:
                state = siam_tracker.track(state, im, name=image_file)

            mask = state['mask']

            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            polygon = state['polygon']

            if not polygon is None:
                polygon = [polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1], polygon[2][0], polygon[2][1], polygon[3][0], polygon[3][1]]
                polygon = np.array(polygon)
                # b_overlap2 = poly_iou(gt[f], polygon)
            else:
                x1, y1, w, h = location
                x2, y2 = x1 + w, y1 + h
                polygon = np.array([x1, y1, x2, y1, x2, y2, x1, y2])

            if poly_iou(np.array(location), np.array(polygon)) > state['choose_thr']:
                record = polygon
                # b_overlaps3.append(b_overlap2)
            else:
                x1, y1, w, h = location
                x2, y2 = x1 + w, y1 + h
                polygon = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
                record = polygon
                # b_overlaps3.append(b_overlap)

            # print('b_overlap: {}, b_overlap2: {}'.format(b_overlap, b_overlap2))
            # b_overlaps.append(b_overlap)
            # b_overlaps2.append(b_overlap2)

            if not 'VOT' in benchmark_name:  # change polygon to [x, y, w, h]
                x1, y1, x2, y2 = record[0], record[1], record[4], record[5]
                record = np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

            if b_overlap > 0:
                regions.append(record)
            else:
                regions.append(2)
                start_frame = f + 5
                lost += 1

            if args.vis:
                COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
                mask = COLORS[mask]
                output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
                cv2.imshow("mask", output)
                cv2.waitKey(1)

        toc += cv2.getTickCount() - tic

    # print('b_overlap: {}, b_overlap2: {}, b_overlap3: {}'.format(np.array(b_overlaps).mean(), np.array(b_overlaps2).mean(), np.array(b_overlaps3).mean()))

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        elif 'OTB' in args.dataset or 'LASOT' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif 'VISDRONE' in args.dataset or 'GOT10K' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))



def main():
    print('Warning: this is a demo to test OceanPlus')
    print('Warning: if you want to test it on VOT2020, please use our integration scripts')
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.online = args.online
    info.TRT = 'TRT' in args.arch

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.vis = args.vis
    siam_tracker = OceanPlus(siam_info)

    if args.mms == 'True':
        MMS = True
    else:
        MMS = False
    siam_net = models.__dict__[args.arch](online=args.online, mms=MMS)
    print('===> init Siamese <====')
    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    siam_net = siam_net.cuda()

    # if info.TRT:
    #     print('===> load model from TRT <===')
    #     print('===> please ignore the warning information of TRT <===')
    #     trtNet = reloadTRT()
    #     siam_net.tensorrt_init(trtNet)

    if args.online:
        online_tracker = ONLINE(info)
    else:
        online_tracker = None

    print('====> warm up <====')
    for i in tqdm(range(20)):
        siam_net.template(torch.rand(1, 3, 127, 127).cuda(), torch.rand(1, 127, 127).cuda())
        siam_net.track(torch.rand(1, 3, 255, 255).cuda())

    # prepare video
    print('====> load dataset <====')
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # hyper-parameters in or not
    if args.hp is None:
        hp = None
    elif isinstance(args.hp, str):
        f = open(join('tune', args.hp), 'r')
        hp = json.load(f)
        f.close()
        print('====> tuning hp: {} <===='.format(hp))
    else:
        raise ValueError('not supported hyper-parameters')

    # tracking all videos in benchmark
    for video in video_keys:
        if args.dataset in ['DAVIS2016', 'DAVIS2017', 'YTBVOS']:  # VOS
            track_vos(siam_tracker, online_tracker, siam_net, dataset[video], args, hp)
        elif args.dataset in ['VOT2020']:  # VOTS (i.e. VOT2020)
            track(siam_tracker, online_tracker, siam_net, dataset[video], args)
        else:
            track_box(siam_tracker, online_tracker, siam_net, dataset[video], args)


# -------------------
# For tune
# -------------------
def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # scale_step, scale_penalty, scale_lr, window_influence

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                     '_small_size_{:.4f}'.format(hp['small_sz']) +
                                     '_big_size_{:.4f}'.format(hp['big_sz']) +
                                     '_lambda_u_{:.4f}'.format(hp['choose_thr']) +
                                     '_lambda_s_{:.4f}'.format(hp['choose_thr']) +
                                     '_cyclic_thr_{:.4f}'.format(hp['choose_thr']) +
                                     '_choose_thr_{:.4f}'.format(hp['choose_thr']) +
                                     '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                     '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                     '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    elif 'GOT10K' in benchmark_name:
        re_video_path = os.path.join(tracker_path, video['name'])
        if not exists(re_video_path): os.makedirs(re_video_path)
        result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
    else:
        result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('OTB'):
            return tracker_path
        elif benchmark_name.startswith('VOT') or benchmark_name.startswith('GOT10K'):
            return 0
        else:
            print('benchmark not supported now')
            return

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]

    # for rgbt splited test

    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            mask_gt = None

            state = tracker.init(im, target_pos, target_sz, net, online=False, mask=mask_gt, debug=False, hp=hp)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1

            polygon = state['polygon']
            if not polygon is None:
                polygon = [polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1], polygon[2][0], polygon[2][1],
                           polygon[3][0], polygon[3][1]]
                polygon = np.array(polygon)
            else:
                x1, y1, w, h = location
                x2, y2 = x1 + w, y1 + h
                polygon = np.array([x1, y1, x2, y1, x2, y2, x1, y2])

            if poly_iou(np.array(location), np.array(polygon)) > state['choose_thr']:
                record = polygon
            else:
                x1, y1, w, h = location
                x2, y2 = x1 + w, y1 + h
                polygon = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
                record = polygon

            if not 'VOT' in benchmark_name:  # change polygon to [x, y, w, h]
                x1, y1, x2, y2 = record[0], record[1], record[4], record[5]
                record = np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

            if b_overlap > 0:
                regions.append(record)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if 'OTB' in benchmark_name or 'LASOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VISDRONE' in benchmark_name  or 'GOT10K' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if 'OTB' in benchmark_name or 'VIS' in benchmark_name or 'VOT' in benchmark_name or 'GOT10K' in benchmark_name:
        return tracker_path
    else:
        print('benchmark not supported now')


def eao_vot_oceanplus(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    re_path = result_path.split('/')[0]
    tracker = result_path.split('/')[-1]

    # debug
    print('======> debug: results_path')
    print(result_path)
    print(os.system("ls"))
    print(join(realpath(dirname(__file__)), '../dataset'))

    # give abs path to json path
    data_path = join(realpath(dirname(__file__)), '../dataset')
    dataset = VOTDataset(config['benchmark'], data_path)

    dataset.set_tracker(re_path, tracker)
    benchmark = EAOBenchmark(dataset)
    eao = benchmark.eval(tracker)
    eao = eao[tracker]['all']

    return eao

def auc_got10k_oceanplus(tracker, net, config):
    """
    get AUC for GOT10K VAL benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)
    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)
    print(result_path)
    auc = eval_got10k_tune(result_path, config['benchmark'])

    return auc

if __name__ == '__main__':
    main()

