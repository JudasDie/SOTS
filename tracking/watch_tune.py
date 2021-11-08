# -*- coding:utf-8 -*-
# ! ./usr/bin/env python

import os
import json
import shutil
import argparse
import numpy as np
import pdb


parser = argparse.ArgumentParser(description='Analysis siamfc tune results')
parser.add_argument('--path', default='./TPE_results/tpe_tune', help='tune result path')
parser.add_argument('--dataset', default='OTB2015', help='test dataset')
parser.add_argument('--model', default='SiamDW', help='model name')


def collect_results(args):
    dirs = os.listdir(args.path)
    print('[*] ===== total {} files in TPE dir'.format(len(dirs)))

    count, scores = 0, []
    if args.model in ['SiamFC', 'SiamDW']:
        scale_penalty, scale_step, lr, window_influence = [], [], [], []
    elif args.model in ['Ocean', 'AutoMatch']:
        penalty_k, lr, window_influence, small_sz, big_sz = [], [], [], [], []
    else:
        pass  # TODO, add other models

    for d in dirs:
        param_path = os.path.join(args.path, d)
        json_path = os.path.join(param_path, 'result.json')

        if not os.path.exists(json_path):
            continue

        try:
            js = json.load(open(json_path, 'r'))
        except:
            continue

        if not "score" in list(js.keys()):
            continue
        elif args.model in ['SiamFC', 'SiamDW']:
            count += 1
            scores.append(js['score'])
            temp = js['config']
            lr.append(temp["lr"])
            window_influence.append(temp["window_influence"])
            scale_step.append(temp["scale_step_FC"])
            scale_penalty.append(temp["scale_penalty_FC"])
        elif args.model in ['Ocean', 'AutoMatch']:
            count += 1
            scores.append(js['score'])
            temp = js['config']
            lr.append(temp["lr"])
            window_influence.append(temp["window_influence"])
            penalty_k.append(temp["penalty_k"])
            small_sz.append(temp["small_sz"])
            big_sz.append(temp["big_sz"])
        else:
            pass # TODO, add other models
 
            
    # find max
    print('{} params group  have been tested'.format(count))
    score = np.array(scores)
    max_idx = np.argmax(score)
    max_score = score[max_idx]

    if args.model in ['SiamFC', 'SiamDW']:
        print('[MODEL]: {}, [DATASET]: {}, scale_penalty: {:.4f}, lr: {:.4f}, window influence: {:.4f}, scale_step: {}, score: {}'.format(args.model, args.dataset, scale_penalty[max_idx], lr[max_idx], window_influence[max_idx], scale_step[max_idx], max_score))
    elif args.model in ['Ocean', 'AutoMatch']:
        print('[MODEL]: {}, [DATASET]: {}, penalty_k: {:.4f}, lr: {:.4f}, window influence: {:.4f}, small_sz: {}, big_sz: {}, score: {}'.format(args.model, args.dataset, penalty_k[max_idx], lr[max_idx], window_influence[max_idx], small_sz[max_idx], big_sz[max_idx], max_score))
    else:
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    collect_results(args)
