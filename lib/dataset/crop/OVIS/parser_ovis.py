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

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default='/datassd/segmentation/OVIS', help='your vid data dir')
args = parser.parse_args()

ovis_base_path = args.dir
sub_sets = sorted({'train'})

ovis = []

for sub_set in sub_sets:
    json_path = join(ovis_base_path, 'annotations_{}.json'.format(sub_set))
    records = json.load(open(json_path, 'r'))
    sub_set_base_path = join(ovis_base_path, 'Images')
    # videos = sorted(listdir(sub_set_base_path))

    # assert len(json) == len(videos)

    s = []
    video_length = []

    videos = records['videos']
    annos = records['annotations']

    for anno in annos:
        v_id = anno['video_id']
        video_info = videos[v_id-1]
        assert v_id == video_info['id']
        video_name = video_info['file_names'][0].split('/')[0]

        v = dict()
        v['base_path'] = join(sub_set_base_path, video_name)
        v['frame'] = []
        # video_base_path = join(sub_set_base_path, video_name)
        gts = anno['bboxes']   # [x,y,w,h] list
        # gts_file = open(gts_path, 'r')
        # gts = gts_file.readlines()
        # gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')

        # get image size
        # im_path = join(video_base_path, '00000001.jpg')
        # im = cv2.imread(im_path)
        # size = im.shape  # height, width
        frame_sz = [anno['width'], anno['height']]  # width,height

        # get all im name
        jpgs = video_info['file_names']

        assert len(jpgs) == len(gts)

        f = dict()
        for idx, img_path in enumerate(jpgs):
            f['frame_sz'] = frame_sz
            f['img_path'] = img_path.split('/')[-1]

            gt = gts[idx]
            if gt is None:
                continue

            bbox = [int(g) for g in gt]   # (x,y,w,h)
            f['bbox'] = bbox
            f['object_id'] = anno['id']

            v['frame'].append(f.copy())
        s.append(v)

        print('[*]subset: {} [*]video id: {:04d} / {:04d} [*]object id : {} [*]frames : {}'.format(sub_set, v_id, len(videos),
                                                                                    anno['id'], len(v['frame'])))

    ovis.append(s)

print('save json (raw ovis info), please wait 1 min~')
json.dump(ovis, open('ovis.json', 'w'), indent=4, sort_keys=True)
print('ovis.json has been saved in ./')
