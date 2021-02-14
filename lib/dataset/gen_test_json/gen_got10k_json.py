# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'
# __description__ = 'generate testing jsons files for GOT10KTEST (no labeled GT, use the toppest in Leadboard)'

import os
import json
from tqdm import tqdm
import numpy as np

videos = {}

path = './GOT10k735'
ori_path = '../../../dataset/GOT10K/test/'
videos = sorted(os.listdir(path))


data= {}
for video in videos:
    gt_path = os.path.join(path, video, '{}_001.txt'.format(video))
    gt = np.loadtxt(gt_path, delimiter=',')
    gt_rect = [list(g) for g in gt]
    data[video] = {}
    data[video]['gt_rect'] = gt_rect
    data[video]['init_rect'] = gt_rect[0]
    imgs = os.listdir(os.path.join(ori_path, video))
    imgs.remove('groundtruth.txt')
    data[video]['image_files'] = sorted(imgs)


# write to json
f = open('GOT10KTEST.json', 'w')
f.write(json.dumps(data))
f.close()

    
