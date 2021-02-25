# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'
# __description__ = 'generate testing jsons files for benchmark xx (prepare for CVPR2021)'

import os
import json
from tqdm import tqdm
import numpy as np

videos = {}

path = '/home/zpzhang/data/testing/TC128'
videos = sorted(os.listdir(path))

data= {}
for video in videos:
    gt_path = os.path.join(path, video, '{}_gt.txt'.format(video))

    if not video in ['Jogging2']:
        frames_path = os.path.join(path, video, '{}_frames.txt'.format(video))
    else:
        frames_path = os.path.join(path, video, '{}_frames.txt'.format(video.lower()))
    gt = np.loadtxt(gt_path, delimiter=',')
    gt_rect = [list(g) for g in gt]
    data[video] = {}
    data[video]['gt_rect'] = gt_rect
    data[video]['init_rect'] = gt_rect[0]

    imgs = sorted(os.listdir(os.path.join(path, video, 'img')))
    imgs = [im for im in imgs if im.endswith('jpg') or im.endswith('png')]

    frames_index = open(frames_path, 'r').readlines()
    frames_index = eval(frames_index[0])
    start, end = frames_index[0], frames_index[1]

    data[video]['image_files'] = ['%04d.jpg'%(i) for i in range(start, end+1)]
    assert len(data[video]['image_files']) == len(data[video]['gt_rect']), video

# write to json
f = open('TC128.json', 'w')
f.write(json.dumps(data))
f.close()

    
