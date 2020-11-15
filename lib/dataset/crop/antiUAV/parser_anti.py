# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import cv2
import json
import glob
import numpy as np
from os.path import join
from os import listdir

import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default='/home/zpzhang/data/training/AntiUAV', help='your vid data dir')
args = parser.parse_args()

anti_base_path = args.dir
# sub_sets = sorted({'train', 'val'})

anti = []

videos = sorted(listdir(anti_base_path))
s = []
for vi, video in enumerate(videos):
    # pdb.set_trace()
    print('video id: {:04d} / {:04d}'.format(vi, len(videos)))
    v = dict()
    v['base_path'] = video
    v['frame'] = []
    video_base_path = join(anti_base_path, video)

    IRannos = json.load(open(join(video_base_path, 'IR_label.json')))

    exists = IRannos['exist']
    gt_rect = IRannos['gt_rect']

    # assert np.array(exists).sum() == len(gt_rect)

    # get image size
    im_path = join(video_base_path, 'IR', '00001.jpg')
    im = cv2.imread(im_path)
    size = im.shape  # height, width
    frame_sz = [size[1], size[0]]  # width,height

    # get all im name
    jpgs = sorted(glob.glob(join(video_base_path, 'IR', '*.jpg')))
    
    f = dict()
    for idx, img_path in enumerate(jpgs):
        if exists[idx] == 1:
            f['frame_sz'] = frame_sz
            f['img_path'] = img_path.split('/')[-1]

            gt = gt_rect[idx]
            bbox = [int(g) for g in gt]   # (x,y,w,h)
            f['bbox'] = bbox
            v['frame'].append(f.copy())
    s.append(v)
anti.append(s)

print('save json (raw anti info), please wait 1 min~')
json.dump(anti, open('anti.json', 'w'), indent=4, sort_keys=True)
print('anti.json has been saved in ./')
