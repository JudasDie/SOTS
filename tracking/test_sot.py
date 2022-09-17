''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: read files with [.yaml] [.txt]
Data: 2021.6.23
'''
import _init_paths
import os
import cv2
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from os.path import exists, join, dirname, realpath

import tracker.sot_tracker as tracker_builder
import utils.model_helper as loader
import utils.box_helper as boxhelper
import utils.log_helper as recorder
import utils.sot_builder as builder
import utils.read_file as reader
from dataset.benchmark_loader import load_sot_benchmark as datafactory
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='Test SOT trackers')
    parser.add_argument('--cfg', type=str, default='experiments/AutoMatch.yaml', help='yaml configure file name')
    parser.add_argument('--resume',  default=None, help='resume checkpoin, if None, use resume in config or epoch testing')
    parser.add_argument('--dataset',  default=None, help='evaluated benchmark, if None, use that in config')
    parser.add_argument('--vis', default=False, type=bool, help='visualization')
    parser.add_argument('--video_path', default=None, help='whether run on a single video (.mp4 or others)')

    args = parser.parse_args()

    return args


def track(inputs):
    siam_tracker, siam_net, video_info, args, config = inputs['tracker'], inputs['network'], inputs['video_info'], \
                                               inputs['args'], inputs['config']

    start_frame, lost, toc, boxes, times  = 0, 0, 0, [], []

    # save result to evaluate
    result_path, time_path = recorder.sot_benchmark_save_path(config, args, video_info)
    if os.path.exists(result_path):
        return  # for mult-gputesting

    image_files, gt = video_info['image_files'], video_info['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if config.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = boxhelper.get_axis_aligned_bbox(gt[f])
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

            init_inputs = {'image': im, 'pos': target_pos, 'sz': target_sz, 'model': siam_net}
            if config.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
                init_inputs['phrase'] = video_info['phrase']
                cand = config.MODEL.CAND if config.MODEL.CAND != 'None' else [None] * 4
                init_inputs['nas_list_z'] = cand[0]
                init_inputs['nas_list_x'] = cand[1]
                init_inputs['nas_list_nlp'] = cand[-2:]
            siam_tracker.init(init_inputs)  # init tracker

            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            boxes.append(1 if 'VOT' in config.TEST.DATA else gt[f])

            times.append(0.02)  # just used for testing on online saver which requires time recording, e.g. got10k

        elif f > start_frame:  # tracking
            state = siam_tracker.track(im)

            location = boxhelper.cxy_wh_2_rect(state['pos'], state['sz'])
            b_overlap = boxhelper.poly_iou(gt[f], location) if 'VOT' in config.TEST.DATA else 1
            times.append(0.02)
            if b_overlap > 0:
                boxes.append(location)
            else:
                boxes.append(2)
                start_frame = f + 5
                lost += 1
        else:
            boxes.append(0)

        toc += cv2.getTickCount() - tic

    save_inputs = {'boxes': boxes, 'times': times, 'result_path': result_path, 'time_path': time_path, 'args': args, 'config': config}
    recorder.sot_benchmark_save(save_inputs)

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video_info['name'], toc, f / toc, lost))


def main():
    args = parse_args()
    if args.cfg is not None:
        config = edict(reader.load_yaml(args.cfg))
    else:
        raise Exception('Please set the config file for tracking test!')
    
    # prepare video
    if args.dataset is None:
        dataset_loader = datafactory(config.TEST.DATA)
    else:
        config.TEST.DATA = args.dataset
        dataset_loader = datafactory(args.dataset)
    dataset = dataset_loader.load()
    video_keys = list(dataset.keys()).copy()
    
    if 'Siam' in config.MODEL.NAME or config.MODEL.NAME in ['Ocean', 'OceanPlus', 'AutoMatch', 'TransT', 'CNNInMo', 'TransInMo', 'VLT_SCAR', 'VLT_TT']:
        siam_tracker = tracker_builder.SiamTracker(config)
        siambuilder = builder.Siamese_builder(config)
        siam_net = siambuilder.build()
        if config.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
            siam_net.backbone.nas(nas_ckpt_path=config.MODEL.NAS_CKPT_PATH)
            siam_net.backbone.load_nlp()
    else:
        raise Exception('Not implemented model type!')

    print(siam_net)
    print('===> init Siamese <====')
    if args.resume is None or args.resume == 'None':
        resume = config.TEST.RESUME
    else:
        resume = args.resume

    if config.MODEL.NAME == 'AutoMatch':
        siam_net = loader.load_pretrain(siam_net, resume, addhead=True, print_unuse=False)
    else:
        siam_net = loader.load_pretrain(siam_net, resume, print_unuse=False)
    siam_net.eval()
    siam_net = siam_net.cuda()


    for video in video_keys:
        inputs = {'tracker': siam_tracker, 'network': siam_net, 'video_info': dataset[video], 'args': args, 'config': config}
        track(inputs)




if __name__ == '__main__':
    main()

