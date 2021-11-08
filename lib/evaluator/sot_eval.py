''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: eval SOT benchmarks (success and precision based), support "OTB, LASOT, GOT10KVAL, TNL2K, TOTB, TREK, NFS, TC128"
Data: 2021.6.23
'''
import sys
import json
import os
import glob
import numpy as np
from tqdm import tqdm
from os.path import join, realpath, dirname

class eval_sot():
    def __init__(self, cfg=None):
        super(eval_sot, self).__init__()
        self.cfg = cfg
        # self.model_name = cfg['MODEL']['NAME'] + '*'
        self.support = ['OTB2015', 'LASOT', 'LASOTTEST', 'GOT10KVAL', 'GOT10KTEST', 'TNL2K', 'TOTB', 'TREK', 'NFS', 'TC128', 'NFS30', 'NFS240', 'UAV123']

    def run(self, dataset='TNL2K', result_path='./result/', tune=False, tracker_reg='Ocean*', start=0, end=1e6, trackers=None):
        '''
        run eval scripts
        '''

        if tune: tracker_reg = dataset

        self.dataset = dataset
        if not dataset in self.support:
            raise Exception('not supported dataset for evaluation!')

        list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
        annos = json.load(open(list_path, 'r'))
        seqs = list(annos.keys())  # dict to list for py3
        n_seq = len(seqs)
        thresholds_overlap = np.arange(0, 1.05, 0.05)
        #thresholds_error = np.arange(0, 51, 1)   

        if not tune:
            if trackers is not None:
                if isinstance(trackers, str):
                    trackers = [trackers]

                trackers = [join(result_path, dataset, tr) for tr in trackers]
            else:
                trackers = glob.glob(join(result_path, dataset, tracker_reg))
                trackers = trackers[start:min(end, len(trackers))]
            success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
            #success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))
        else:
            success_overlap = np.zeros((n_seq, 1, len(thresholds_overlap)))
            #success_error = np.zeros((n_seq, 1, len(thresholds_error)))
        for i in tqdm(range(n_seq)):
            seq = seqs[i]
            if seq in ['advSamp_Baseball_game_002-Done', 'advSamp_monitor_bikeyellow']:  # TNL2K wrong annotations (first version)
                gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)[:-1, :]
            else:
                gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)

            gt_center = self.convert_bb_to_center(gt_rect)

            if not tune:
                for j in range(len(trackers)):
                    tracker = trackers[j]
            #        print('{:d} processing:{} tracker: {}'.format(i, seq, tracker))
                    bb = self.get_result_bb(tracker, seq)
                    center = self.convert_bb_to_center(bb)
                    success_overlap[i][j] = self.compute_success_overlap(gt_rect, bb)
            #        success_error[i][j] = self.compute_success_error(gt_center, center)
            else:
                bb = self.get_result_bb_tune(result_path, seq)
                center = self.convert_bb_to_center(bb)
                success_overlap[i][0] = self.compute_success_overlap(gt_rect, bb)
            #    success_error[i][0] = self.compute_success_error(gt_center, center)


        max_auc = 0.
        max_name = ''

        tracker_len = len(trackers) if not tune else 1

        for i in range(tracker_len):
            auc = success_overlap[:, i, :].mean()
            #prec = success_error[:, i, :].mean()
            if auc > max_auc:
                max_auc = auc
                #max_prec = prec
                max_name = trackers[i] if not tune else 'Tune'

            if not tune:
                print('%s(%.4f)' % (trackers[i], auc))

       # print('\n%s Best: %s(AUC/Prec:%.4f/%.4f)' % (dataset, max_name, max_auc, max_prec))
        print('\n%s Best AUC: %s %.4f' % (dataset, max_name, max_auc))

        return max_auc


    def overlap_ratio(self, rect1, rect2):
        '''
        Compute overlap ratio between two rects
        - rect: 1d array of [x,y,w,h] or
                2d array of N x [x,y,w,h]
        '''

        if rect1.ndim == 1:
            rect1 = rect1[None, :]
        if rect2.ndim == 1:
            rect2 = rect2[None, :]

        left = np.maximum(rect1[:, 0], rect2[:, 0])
        right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
        top = np.maximum(rect1[:, 1], rect2[:, 1])
        bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

        intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
        iou = np.clip(intersect / union, 0, 1)
        return iou


    def compute_success_overlap(self, gt_bb, result_bb):
        '''
        compute success overlap with interval of 0.05 (iou)
        '''
        thresholds_overlap = np.arange(0, 1.05, 0.05)
        n_frame = len(gt_bb)
        success = np.zeros(len(thresholds_overlap))
        iou = self.overlap_ratio(gt_bb, result_bb)
        for i in range(len(thresholds_overlap)):
            success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
        return success


    def compute_success_error(self, gt_center, result_center):
        '''
        compute success overlap with interval of 1 (pixel)
        '''
        thresholds_error = np.arange(0, 51, 1)
        n_frame = len(gt_center)
        success = np.zeros(len(thresholds_error))
        dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
        for i in range(len(thresholds_error)):
            success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
        return success


    def get_result_bb(self, arch, seq):
        '''
        parser result file in .txt format
        '''
        if not 'GOT10K' in self.dataset:
            result_path = join(arch, seq + '.txt')
        else:
            result_path = join(arch, seq, seq + '.txt')
        temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
        return np.array(temp)


    def get_result_bb_tune(self, result_path, seq):
        '''
        parser result file in .txt format
        '''
        result_path = join(result_path, seq + '.txt')
        temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
        return np.array(temp)

    def convert_bb_to_center(self, bboxes):
        '''
        computer box center
        x1y1wh --> cxcy
        '''
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

if __name__ == "__main__":
    evaler = eval_sot()

    if len(sys.argv) < 5:
           print('python ./lib/evaluator/eval_sot.py TNL2K ./result Ocean* 0 1000')
           exit()

    dataset = sys.argv[1]
    result_path = sys.argv[2]
    tracker_reg = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    evaler.run(dataset, result_path, tracker_reg, start, end)


