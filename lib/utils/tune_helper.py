''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: build sot models (siamese)
Data: 2021.6.23
'''

# import importlib
import os
import cv2
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from easydict import EasyDict as edict
from os.path import exists, join, dirname, realpath

import utils.model_helper as loader
import utils.box_helper as boxhelper
import utils.log_helper as recorder
import utils.read_file as reader

from evaluator.sot_eval import eval_sot
from dataset.benchmark_loader import load_sot_benchmark as datafactory
from evaluator.vot_eval.pysot.datasets import VOTDataset
from evaluator.vot_eval.pysot.evaluation import EAOBenchmark

class SOTtuner_builder(nn.Module):
    def __init__(self, cfg):
        super(SOTtuner_builder).__init__()
        self.cfg = cfg
        self.dataset = self.cfg.TUNE.DATA
        self.evaler = eval_sot()

    def run(self, inputs):
        tracker, net, hp = inputs['tracker'], inputs['model'], inputs['config']
        if 'VOT' not in self.dataset:
            return self.eval_ope(tracker, net, hp['hp'])
        else:
            return self.eval_vot(tracker, net, hp['hp'])

    def eval_ope(self, tracker, net, hp):
        """
        eval ope (one pass evaluation benchmarks, e.g. OTB2015, GOT10K and so on)
        """
        dataset_loader = datafactory(self.dataset)
        dataset = dataset_loader.load()
        video_keys = list(dataset.keys()).copy()
        random.shuffle(video_keys)

        for video in video_keys:
            result_path = self.track_tune(tracker, net, dataset[video], hp)

        auc = self.evaler.run(dataset=self.dataset, result_path=result_path, tune=True)

        return auc


    def eval_vot(self, tracker, net, hp):
        """
        eval vot (before VOT2020 is supported)
        """
        dataset_loader = datafactory(self.dataset)
        dataset = dataset_loader.load()
        video_keys = sorted(list(dataset.keys()).copy())

        for video in video_keys:
            result_path = self.track_tune(tracker, net, dataset[video], hp)

        re_path = result_path.split('/')[0]
        tracker = result_path.split('/')[-1]

        # debug
        # print('======> debug: results_path')
        # print(result_path)
        # print(os.system("ls"))
        # print(join(realpath(dirname(__file__)), '../dataset'))

        # give abs path to json path
        data_path = os.path.join(realpath(dirname(__file__)), '../../dataset')
        dataset = VOTDataset(self.dataset, data_path)

        dataset.set_tracker(re_path, tracker)
        benchmark = EAOBenchmark(dataset)
        eao = benchmark.eval(tracker)
        eao = eao[tracker]['all']

        return eao

    def track_tune(self, siam_tracker, siam_net, video, hp):
        resume = self.cfg.TUNE.RESUME.split('/')[-1].split('.')[0]

        if self.cfg.MODEL.NAME in ['Ocean', 'AutoMatch']:
            tracker_path = os.path.join('test', (self.dataset + resume +
                                         '_small_size_{:.4f}'.format(hp['small_sz']) +
                                         '_big_size_{:.4f}'.format(hp['big_sz']) +
                                         '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                         '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                         '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
        elif self.cfg.MODEL.NAME in ['SiamFC', 'SiamDW']:
            tracker_path = os.path.join('test', (self.dataset + resume +
                                         '_scale_step_{:.4f}'.format(hp['scale_step_FC']) +
                                         '_p_{:.4f}'.format(hp['scale_penalty_FC']) +
                                         '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                         '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
        else:
            raise ValueError('not implemented model')


        if not os.path.exists(tracker_path):
            os.makedirs(tracker_path)

        if 'VOT' in self.dataset:
            baseline_path = os.path.join(tracker_path, 'baseline')
            video_path = os.path.join(baseline_path, video['name'])
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, video['name'] + '_001.txt')
        # elif 'GOT10K' in self.dataset:
        #     re_video_path = os.path.join(tracker_path, video['name'])
        #     if not exists(re_video_path): os.makedirs(re_video_path)
        #     result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
        else:
            result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

        # occ for parallel running
        if not os.path.exists(result_path):
            fin = open(result_path, 'w')
            fin.close()
        else:
            if self.dataset.startswith('OTB'):
                return tracker_path
            elif self.dataset.startswith('VOT') or self.dataset.startswith('GOT10K'):
                return 0
            else:
                print('benchmark not supported now')
                return

        start_frame, lost_times, toc, boxes = 0, 0, 0, []

        image_files, gt = video['image_files'], video['gt']

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            if f == start_frame:  # init
                cx, cy, w, h = boxhelper.get_axis_aligned_bbox(gt[f])
                target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

                init_inputs = {'image': im, 'pos': target_pos, 'sz': target_sz, 'model': siam_net}
                siam_tracker.init(init_inputs, hp=hp)

                boxes.append(1 if 'VOT' in self.dataset else gt[f])

            elif f > start_frame:
                state = siam_tracker.track(im)
                location = boxhelper.cxy_wh_2_rect(state['pos'], state['sz'])
                b_overlap = boxhelper.poly_iou(gt[f], location) if 'VOT' in self.dataset else 1
                if b_overlap > 0:
                    boxes.append(location)
                else:
                    boxes.append([float(2)])
                    lost_times += 1
                    start_frame = f + 5  # skip 5 frames
            else:  # skip
                boxes.append([float(0)])

        # save results for OTB
        if 'OTB' in self.dataset or 'LASOT' in self.dataset:
            with open(result_path, "w") as fin:
                for x in boxes:
                    p_bbox = x.copy()
                    fin.write(
                        ','.join(
                            [str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif 'VISDRONE' in self.dataset or 'GOT10K' in self.dataset or 'NFS' in self.dataset or 'TC128' in self.dataset or 'UAV123' in self.dataset:
            with open(result_path, "w") as fin:
                for x in boxes:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif 'VOT' in self.dataset:
            with open(result_path, "w") as fin:
                for x in boxes:
                    if isinstance(x, int):
                        fin.write("{:d}\n".format(x))
                    else:
                        p_bbox = x.copy()
                        fin.write(','.join([str(i) for i in p_bbox]) + '\n')

        if 'OTB' in self.dataset or 'VIS' in self.dataset or 'VOT' in self.dataset or 'GOT10K' in self.dataset or 'NFS' in self.dataset or 'TC128' in self.dataset or 'UAV123' in self.dataset:
            return tracker_path
        else:
            print('benchmark not supported now')
