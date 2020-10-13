# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'
# __description__ = 'generate testing jsons files for benchmark xx (prepare for CVPR2021)'

import os
import json
from tqdm import tqdm
import numpy as np

videos = {}

path = '/data/share/NLPTracking'
videos = sorted(os.listdir(path))

data= {}
for video in videos:
    gt_path = os.path.join(path, video, 'groundtruth.txt')
    gt = np.loadtxt(gt_path, delimiter=',')
    gt_rect = [list(g) for g in gt]
    data[video] = {}
    data[video]['gt_rect'] = gt_rect
    data[video]['init_rect'] = gt_rect[0]
    data[video]['image_files'] = sorted(os.listdir(os.path.join(path, video, 'imgs')))


# write to json
f = open('tn.json', 'w')
f.write(json.dumps(data))
f.close()

    
