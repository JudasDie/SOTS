''' Details
Author: Zhipeng Zhang/Chao Liang
Function: test MOT method CSTrack
Data: 2022.4.7
'''

import _init_paths

import os
import os.path as osp
import cv2
import logging
import argparse
import numpy as np
import torch
import pickle
import motmetrics as mm

import dataset.benchmark_loader as benchmark_loader
import utils.read_file as reader
from loguru import logger
from utils.log_helper import Timer
from evaluator.mot_eval import Evaluator
from utils.log_helper import mot_benchmark_save
from utils.vis_helper import plot_mot_tracking, plot_mot_tracking_online
from utils.tracking_helper import update_cstrack_hypers, get_mot_benchmark_path

import tracker.mot_tracker as tracker_builder

from easydict import EasyDict as edict
import pdb

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')

    # test data
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results', help='where to save the results')
    parser.add_argument('--benchmark', type=str, default=None, help='None or key in yaml, None=COMMOM_HYPES')
    parser.add_argument('--val_mot15', default=False, help='val mot15')
    parser.add_argument('--test_mot15', default=False, help='test mot15')

    parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
    parser.add_argument('--test_mot16', default=False, help='test mot16')

    parser.add_argument('--val_mot17', default=False, help='val mot17')
    parser.add_argument('--test_mot17', default=False, help='test mot17')

    parser.add_argument('--val_mot20', default=False, help='val mot20')
    parser.add_argument('--test_mot20', default=False, help='test mot20')

    # MOT Challenges website requires submit both val amd test results,
    # set all_motx to True if you want to test both val and test sets

    parser.add_argument('--val_hf', type=int, default=0, help='val_hf')

    # hyper-parameters
    parser.add_argument('--cfg', type=str, default=None, help='model.yaml path')
    parser.add_argument('--nms_thres', type=float, default=None, help='iou thresh for nms')
    parser.add_argument('--conf_thres', type=float, default=None, help='object confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=None, help='tracking buffer')
    parser.add_argument('--min_box_area', type=float, default=None, help='filter out tiny boxes')
    parser.add_argument('--img_size', type=tuple, default=None, help='input image size')
    parser.add_argument('--mean', type=list, default=None, help='image normalize mean')
    parser.add_argument('--std', type=list, default=None, help='image normalize std')


    # model and weights
    parser.add_argument('--weights', type=str, default=None, help='model.pt path(s)')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset, e.g. mot')

    # GPUs
    parser.add_argument('--device', default='0', help='-1 for CPU, use comma for multiple gpus, e.g. 0,1')

    # vis
    parser.add_argument('--save_videos', type=bool, default=False, help='save video or not')
    parser.add_argument('--vis_state', type=bool, default=False, help='vis feature during tracking')
    parser.add_argument('--vis', type=bool, default=False, help='vis feature during tracking')

    args = parser.parse_args()

    return args


def run_seq(opt, dataloader, result_filename, seq):
    # build tracker
    logger.info('build tracker ...')
    if opt.cfg.MODEL.Name in ['JDE', 'CSTrack', 'OMC']:
        tracker = tracker_builder.JDETracker(opt, frame_rate=opt.frame_rate)
    else:
        raise ValueError('tracker {} is not supported'.format(opt.cfg.MODEL.Name))

    timer, results, frame_id, seq_num, once, horizontal_list = Timer(), [], 0, 0, 1, {}
    vis_window_name = 'Seq: {0}-{1}, Tracker: {2}'.format(opt.benchmark_name, seq, opt.MODEL.Name)

    for path, img, img0, frame_id_start in dataloader:
        if once == 1:
            frame_id = frame_id_start
            once = 0

        if frame_id % 20 == 0 and not frame_id == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0, seq, opt)   # no init process for MOT

        # record and update
        online_tlwhs, online_ids = [], []   # (x,y,w,h), id

        for t in online_targets:
            tlwh, tid = t.tlwh, t.track_id  # (x,y,w,h), id

            horizontal = tlwh[2] / tlwh[3] > 0.6  # people are not standing  # TODO: this setting only supports  pedestrian tracking
            horizontal_add = round(tlwh[2] / tlwh[3], 1)                     # TODO: for other classes, e.g. cars, you should change here
            if horizontal_add not in horizontal_list:
                horizontal_list[horizontal_add] = 0
            horizontal_list[horizontal_add] += 1

            if tlwh[2] * tlwh[3] > opt.min_box_area and not horizontal:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        timer.toc()

        if opt.args.vis and online_tlwhs:
            plot_mot_tracking_online(img0, online_tlwhs, online_ids, frame_id=frame_id, name=vis_window_name, seq_name=seq, opt=opt)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        frame_id += 1

    # save results
    result_detection, id_num = mot_benchmark_save(result_filename, results, opt.data_type)   # total detections, id summary
    logger.info('horizontal peoples: '.format(horizontal_list))

    return frame_id, timer.average_time, timer.calls, result_detection, id_num, frame_id_start


def main(opt, data_root='/data/MOT16/train', seqs=('MOT16-05',), exp_name='demo'):
    result_root = osp.join(opt.args.output_dir, exp_name, 'result_{}'.format(opt.benchmark_name))
    if not osp.exists(result_root): os.makedirs(result_root)
    vis_video_root = os.path.join(opt.args.output_dir, exp_name, 'vis/video')
    if not osp.exists(vis_video_root): os.makedirs(vis_video_root)
    vis_img_root = os.path.join(opt.args.output_dir, exp_name, 'vis/img')
    if not osp.exists(vis_img_root): os.makedirs(vis_img_root)
    opt.vis_img_root = vis_img_root

    env_yaml_path = osp.join(opt.args.output_dir, exp_name, opt.cfg.MODEL.Name + '_conda.yaml')
    logger.info('save conda/python environments to {}'.format(env_yaml_path))
    os.system('conda env export > {}'.format(env_yaml_path))
    logger_path = osp.join(opt.args.output_dir, exp_name, 'test_{}.log'.format(opt.benchmark_name))
    logger.add(logger_path)
    logger.info('log is saved to {}'.format(logger_path))

    # record
    accs, timer_avgs, timer_calls, result_detection, result_id, n_frame = [], [], [], [], [], 0

    # run tracking
    for seq in seqs:
        logger.info('testing seq: {} ...'.format(seq))
        if opt.benchmark_name in ['MOT15', 'MOT16', 'MOT17', 'MOT20']:
            img_path = osp.join(data_root, seq, 'img1')
            opt.data_type = 'MOTChallenge'
        else:
            img_path = osp.join(data_root, seq)

        dataloader = benchmark_loader.load_mot_benchmark(img_path, opt.img_size, opt.args.val_hf)

        # TODO: only support MOTChallenge Benchmarks format now, check if evaluate other benchmarks
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        opt.frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        logger.info('frame rate: {}'.format(opt.frame_rate))

        # number of frames, time average, timer calls, total detections, id summary, frame_id_start
        nf, ta, tc, detection_num, id_num, frame_id_start = run_seq(opt, dataloader, result_filename, seq)

        # record and summary
        result_detection += [detection_num]
        result_id += [id_num]
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # evaluation
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, opt.data_type)
        accs.append(evaluator.eval_file(result_filename, frame_id_start, nf))

        if opt.args.save_videos:
            output_video_path = osp.join(vis_video_root, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(osp.join(vis_img_root, seq), output_video_path)
            os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    print("detection_num:", result_detection, sum(result_detection))
    print("id_num:", result_id, sum(result_id))
    Evaluator.save_summary(summary, os.path.join(opt.args.output_dir, exp_name, 'summary_{}.xlsx'. format(exp_name)))

    # save opt
    # opt_save_path = osp.join(opt.args.output_dir, 'opt.txt')
    # logger.info('save opt/hyper-parameters to {}'.format(opt_save_path))
    # with open(opt_save_path, 'w') as file:
    #     file.write(pickle.dumps(opt))
    # file.close()


if __name__ == '__main__':

    args = parse_args()
    # # debug -------------------
    # print('debug, remove after debugging')
    # args.data_dir = '/datassd/tracking/MOT/MOTCH'
    # args.exp_name = 'CSTrack_NewCode'
    # args.output_dir = 'results'
    # args.val_mot16 = True
    # args.cfg = '../experiments/CSTrack.yaml'
    # args.nms_thres = 0.6
    # args.conf_thres = 0.5
    # args.track_buffer = 30
    # args.min_box_area = 100
    # args.weights = '../snapshot_rep2/CSTrack/ex0/weights/last.pt'
    # args.device = 0
    # args.vis = True
    # args.save_videos = True
    # # -------------------------

    if args.cfg is not None:
        config = edict(reader.load_yaml(args.cfg))
    else:
        raise Exception('Please set the config file for tracking test!')

    # update hyper-parameters
    logger.info('update hyper-parameters:')
    opt = edict()
    opt = update_cstrack_hypers(opt, args, config)
    logger.info(opt)

    logger.info('visible gpus: {}'.format(opt.args.device))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.args.device)

    # get seq and data root
    seqs, data_root, benchmark_name = get_mot_benchmark_path(opt)
    seqs = [seq.strip() for seq in seqs]
    opt.benchmark_name = benchmark_name

    exp_name = opt.args.exp_name if opt.args.exp_name is not None else benchmark_name
    logger.info('experiment name: {}'.format(exp_name))

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=exp_name)
