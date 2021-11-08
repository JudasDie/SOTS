''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: a simple evaluation for SOT methods
Data: 2021.6.23
'''
import _init_paths
import os
import argparse
from evaluator.sot_eval import eval_sot


def parse_args():
    parser = argparse.ArgumentParser(description='Test SOT Trackers')
    parser.add_argument('--dataset',  default=None, help='evaluation dataset')
    parser.add_argument('--trackers',  default=None, type=str, nargs='+', help='trackers')

    args = parser.parse_args()

    return args

args = parse_args()

print('[*] ====> evaluation <====')

if args.trackers is None:
    trackers = os.listdir(os.path.join('./result', args.dataset))
#    trackers = " ".join(trackers)
else:
    trackers = args.trackers


if 'VOT' in args.dataset:
    print('python lib/evaluator/vot_eval/bin/eval.py --dataset_dir dataset --dataset {0} --tracker_result_dir result/{0} --trackers {1}'.format(args.dataset, trackers))
    os.system('python lib/evaluator/vot_eval/bin/eval.py --dataset_dir dataset --dataset {0} --tracker_result_dir result/{0} --trackers {1}'.format(args.dataset, trackers))
else:
    evaler = eval_sot()
    evaler.run(dataset=args.dataset, result_path='./result/', tracker_reg=trackers[0][:5]+'*', trackers=trackers)
