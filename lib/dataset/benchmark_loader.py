''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: load benchmark dataset for testing
Data: 2021.6.23
'''

import os
import json
import glob
import numpy as np
from os.path import join
from os.path import join, realpath, dirname, exists
import pdb

class load_sot_benchmark():
    def __init__(self, benchmark):
        super(load_sot_benchmark, self).__init__()
        self.dataset = benchmark

    def load(self):
        if 'OTB' in self.dataset and not self.dataset == 'TOTB':
            return self.load_OTB()
        elif 'VOT' in self.dataset:
            return self.load_VOT()
        elif 'TC128' in self.dataset:
            return self.load_TC128()
        elif 'UAV123' in self.dataset:
            return self.load_UAV123()
        elif 'GOT10KVAL' in self.dataset:
            return self.load_GOT10KVAL()
        elif 'GOT10KTEST' in self.dataset:
            return self.load_GOT10KTEST()
        elif 'LASOT' in self.dataset:
            return self.load_LASOT()
        elif 'DAVIS' in self.dataset:
            return self.load_DAVIS()
        elif 'YTBVOS' in self.dataset:
            return self.load_YTBVOS()
        elif 'TNL2K' in self.dataset:
            return self.load_TNL2K()
        elif 'TOTB' in self.dataset:
            return self.load_TOTB()
        elif 'TREK' in self.dataset:
            return self.load_TREK()
        elif 'NFS' in self.dataset:
            return self.load_NFS()
        elif 'TRACKINGNET' in self.dataset:
            return self.load_TRACKINGNET()
        else:
            raise Exception('Not implemented benchmark!')

    def load_OTB(self):
        """
        OTB: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v

        return info

    def load_VOT(self):
        """
        VOT: https://www.votchallenge.net/
        """
        info = {}
        if not 'VOT2020' in self.dataset:
            base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
            list_path = join(base_path, 'list.txt')
            with open(list_path) as f:
                videos = [v.strip() for v in f.readlines()]
            videos = sorted(videos)
            for video in videos:
                video_path = join(base_path, video)
                image_path = join(video_path, '*.jpg')
                image_files = sorted(glob.glob(image_path))
                if len(image_files) == 0:  # VOT2018
                    image_path = join(video_path, 'color', '*.jpg')
                    image_files = sorted(glob.glob(image_path))
                gt_path = join(video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
                info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
        else:
            base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
            list_path = join(base_path, 'list.txt')
            with open(list_path) as f:
                videos = [v.strip() for v in f.readlines()]
            videos = sorted(videos)
            for video in videos:
                video_path = join(base_path, video)
                image_path = join(video_path, '*.jpg')
                image_files = sorted(glob.glob(image_path))
                if len(image_files) == 0:  # VOT2018
                    image_path = join(video_path, 'color', '*.jpg')
                    image_files = sorted(glob.glob(image_path))
                gt_path = join(video_path, 'groundtruth.txt')
                gt = open(gt_path, 'r').readlines()
                info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
        return info

    def load_RGBT234(self):
        """
        RGBT234: https://sites.google.com/view/ahutracking001/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['infrared_imgs'] = [join(base_path, path_name, 'infrared', im_f) for im_f in
                                        info[v]['infrared_imgs']]
            info[v]['visiable_imgs'] = [join(base_path, path_name, 'visible', im_f) for im_f in
                                        info[v]['visiable_imgs']]
            info[v]['infrared_gt'] = np.array(info[v]['infrared_gt'])  # 0-index
            info[v]['visiable_gt'] = np.array(info[v]['visiable_gt'])  # 0-index
            info[v]['name'] = v
        return info

    def load_VISDRONEVAL(self):
        """
        visdrone validation dataset
        VISDRONE: http://aiskyeye.com/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'annotations')
        attr_path = join(base_path, 'attributes')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_VISDRONETEST(self):
        """
        visdrone testing dataset
        VISDRONE: http://aiskyeye.com/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'initialization')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',').reshape(1, 4)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_GOT10KVAL(self):
        """
        GOT10K validation dataset
        GOT10K: http://got-10k.aitestunion.com/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))

        try:
            videos.remove('list.txt')
        except:
            pass

        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_GOT10KTEST(self):
        """
        GOT10K testing dataset
        GOT10K: http://got-10k.aitestunion.com/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}

        return info

    def load_LASOT(self):
        """
        LASOT: https://arxiv.org/abs/1809.07845
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        testingvideos = list(jsons.keys())

        father_videos = sorted(os.listdir(base_path))
        for f_video in father_videos:
            f_video_path = join(base_path, f_video)
            son_videos = sorted(os.listdir(f_video_path))
            for s_video in son_videos:
                if s_video not in testingvideos:  # 280 testing videos
                    continue

                s_video_path = join(f_video_path, s_video)
                # ground truth
                gt_path = join(s_video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',')
                gt = gt - [1, 1, 0, 0]
                # get img file
                img_path = join(s_video_path, 'img', '*jpg')
                image_files = sorted(glob.glob(img_path))

                info[s_video] = {'image_files': image_files, 'gt': gt, 'name': s_video}
        return info

    def load_DAVIS(self):
        """
        DAVIS: https://davischallenge.org/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', 'DAVIS')
        list_path = join(realpath(dirname(__file__)), '../../dataset', 'DAVIS', 'ImageSets', self.dataset[-4:],
                         'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            info[video] = {}
            info[video]['anno_files'] = sorted(glob.glob(join(base_path, 'Annotations/480p', video, '*.png')))
            info[video]['image_files'] = sorted(glob.glob(join(base_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video

        return info

    def load_YTBVOS(self):
        """
        YTBVOS: https://youtube-vos.org/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', 'YTBVOS', 'valid')
        json_path = join(realpath(dirname(__file__)), '../../dataset', 'YTBVOS', 'valid', 'meta.json')
        meta = json.load(open(json_path, 'r'))
        meta = meta['videos']
        info = dict()
        for v in meta.keys():
            objects = meta[v]['objects']
            frames = []
            anno_frames = []
            info[v] = dict()
            for obj in objects:
                frames += objects[obj]['frames']
                anno_frames += [objects[obj]['frames'][0]]
            frames = sorted(np.unique(frames))
            info[v]['anno_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in frames]
            info[v]['anno_init_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in anno_frames]
            info[v]['image_files'] = [join(base_path, 'JPEGImages', v, im_f + '.jpg') for im_f in frames]
            info[v]['name'] = v

            info[v]['start_frame'] = dict()
            info[v]['end_frame'] = dict()
            for obj in objects:
                start_file = objects[obj]['frames'][0]
                end_file = objects[obj]['frames'][-1]
                info[v]['start_frame'][obj] = frames.index(start_file)
                info[v]['end_frame'][obj] = frames.index(end_file)

        return info

    def load_TNL2K(self):
        """
        TNL2K: https://sites.google.com/view/langtrackbenchmark/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        videos = list(jsons.keys())

        for video in videos:
            vinfo = jsons[video]
            # ground truth
            gt = np.array(vinfo['gt_rect'])
            # get img file
            imgs = vinfo['image_files']
            image_files = [join(base_path, video, 'imgs', im) for im in imgs]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_TOTB(self):
        """
        TOTB: https://hengfan2010.github.io/projects/TOTB/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        videos = list(jsons.keys())

        for video in videos:
            vinfo = jsons[video]
            # ground truth
            gt = np.array(vinfo['gt_rect'])
            # get img file
            imgs = vinfo['image_files']
            image_files = [join(base_path, video.split('_')[0], video, 'img', im) for im in imgs]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_TREK(self):
        """
        TREK: https://machinelearning.uniud.it/datasets/trek100/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        videos = list(jsons.keys())

        for video in videos:
            vinfo = jsons[video]
            # ground truth
            gt = np.array(vinfo['gt_rect'])
            # get img file
            imgs = vinfo['image_files']
            image_files = [join(base_path, video, 'img', im) for im in imgs]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_NFS(self):
        """
        NFS: http://ci2cv.net/nfs/index.html
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        videos = list(jsons.keys())

        for video in videos:
            vinfo = jsons[video]
            # ground truth
            gt = np.array(vinfo['gt_rect'])
            # get img file
            imgs = vinfo['img_names']
            image_files = [join(base_path, im) for im in imgs]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_TC128(self):
        """
        TC128: https://www3.cs.stonybrook.edu/~hling/data/TColor-128/TColor-128.html
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        videos = list(jsons.keys())

        for video in videos:
            vinfo = jsons[video]
            # ground truth
            gt = np.array(vinfo['gt_rect'])
            # get img file
            imgs = vinfo['image_files']
            image_files = [join(base_path, video, 'img', im) for im in imgs]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_UAV123(self):
        """
        UAV123: https://cemse.kaust.edu.sa/ivul/uav123
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        videos = list(jsons.keys())

        for video in videos:
            vinfo = jsons[video]
            # ground truth
            gt = np.array(vinfo['gt_rect'])
            # get img file
            imgs = vinfo['img_names']
            image_files = [join(base_path, 'data_seq/UAV123/', im) for im in imgs]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info

    def load_TRACKINGNET(self):
        """
        TrackingNet: https://tracking-net.org/
        """
        info = {}
        base_path = join(realpath(dirname(__file__)), '../../dataset', self.dataset)
        seq_path = join(base_path, 'frames')
        anno_path = join(base_path, 'anno')
        videos = sorted(os.listdir(seq_path))

        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            nums = len(image_files)
            image_files = [join(video_path, '{}.jpg'.format(str(i))) for i in range(nums)]
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',').reshape(1, 4)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        return info
