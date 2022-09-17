import os
import json
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

class NLLaSOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, absent, load_img=False, phrase=None):
        super(NLLaSOTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)
        self.absent = np.array(absent, np.int8)
        self.phrase = phrase

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            # print(traj_file)
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
            else:
                print("File not exists: ", traj_file)
            if self.name == 'monkey-17':
                pred_traj = pred_traj[:len(self.gt_traj)]
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj
        self.tracker_names = list(self.pred_trajs.keys())



class NLLaSOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, load_nlp=False):
        super(NLLaSOTDataset, self).__init__(name, dataset_root)
        dataset_root = os.path.join(dataset_root.split('NLLaSOT')[0], 'LaSOT')
        with open(os.path.join(dataset_root, name.split('NL')[-1]+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        NLLaSOT_list = ['airplane-1','airplane-13','airplane-15','airplane-9','bird-15','bird-17','bird-2','bird-3',
                        'book-10','book-11','book-19','book-3','crab-12','crab-18','crab-3','crab-6','hat-1','hat-18',
                        'hat-2','hat-5','rubicCube-1','rubicCube-14','rubicCube-19','rubicCube-6','tank-14','tank-16',
                        'tank-6','tank-9','umbrella-17','umbrella-19','umbrella-2','umbrella-9','volleyball-1',
                        'volleyball-13','volleyball-18','volleyball-19','yoyo-15','yoyo-17','yoyo-19','yoyo-7',
                        'zebra-10','zebra-14','zebra-16','zebra-17']
        # keys = meta_data.keys()
        pbar = tqdm(NLLaSOT_list, desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)

            for i in range(len(meta_data[video]['img_names'])):
                root_name = meta_data[video]['img_names'][i].split('/')[0].split('-')[0]
                meta_data[video]['img_names'][i] = root_name + '/' + meta_data[video]['img_names'][i]
            if load_nlp:
                self.videos[video] = NLLaSOTVideo(video,
                                                dataset_root,
                                                meta_data[video]['video_dir'],
                                                meta_data[video]['init_rect'],
                                                meta_data[video]['img_names'],
                                                meta_data[video]['gt_rect'],
                                                meta_data[video]['attr'],
                                                meta_data[video]['absent'],
                                                load_img,
                                                meta_data[video]['phrase'])
            else:
                self.videos[video] = NLLaSOTVideo(video,
                                              dataset_root,
                                              meta_data[video]['video_dir'],
                                              meta_data[video]['init_rect'],
                                              meta_data[video]['img_names'],
                                              meta_data[video]['gt_rect'],
                                              meta_data[video]['attr'],
                                              meta_data[video]['absent'],
                                                load_img)

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)


