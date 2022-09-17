''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: load benchmark dataset for testing
Data: 2021.6.23
'''

import os
import cv2
import json
import glob
import numpy as np
from os.path import join
from os.path import join, realpath, dirname, exists
import utils.tracking_helper as image_helper
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
            if '.' in f_video:
                continue
            f_video_path = join(base_path, f_video)
            son_videos = sorted(os.listdir(f_video_path))
            for s_video in son_videos:
                if s_video not in testingvideos:  # 280 testing videos
                    continue

                s_video_path = join(f_video_path, s_video)
                # ground truth
                gt_path = join(s_video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',')
                # gt = gt - [1, 1, 0, 0]
                # get img file
                img_path = join(s_video_path, 'img', '*jpg')
                image_files = sorted(glob.glob(img_path))

                info[s_video] = {'image_files': image_files, 'gt': gt, 'name': s_video, 'phrase': jsons[s_video]['phrase'] if 'phrase' in jsons[s_video].keys() else None}
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



# ------------------------- MOT ----------------------------

class load_mot_benchmark:  # for inference
    """
    load evaluation MOT videos
    supported benchmarks: MOT Challenge
    """
    def __init__(self, path, img_size=(1088, 608), val_hf=False):
        # supported = ['MOT15test', 'MOT15val', 'MOT16test', 'MOT16val', 'MOT17test', 'MOT17val', 'MOT20test', 'MOT20val']
        # if benchmark not in supported: raise ValueError('{0} is not supported for evaluation, '
        #                                                 'pls update codes to support {0}'.format(benchmark))

        """
        load mot benchmark for evaluation
        :param path: video images' path
        :param img_size: input image size for the tracker (width, height)
        :param val_hf:
        return: RGB images
        """
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.frame_id = 0
        if val_hf == 1:
            self.files = self.files[:int(len(self.files)*0.5)+1]
        if val_hf == 2:
            self.frame_id = int(len(self.files)*0.5)+1
            self.files = self.files[int(len(self.files)*0.5)+1:]
        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF: raise StopIteration

        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = image_helper.letterbox_jde(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0, self.frame_id

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = image_helper.letterbox_jde(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF
