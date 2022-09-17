import json
import os
import numpy as np
from tqdm import tqdm

from .dataset import Dataset
from .video import Video


class ITBVideo(Video):
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
            gt_rect, attr, load_img=False, phrase=None):
        super(ITBVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)
        self.phrase = phrase

class ITBDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, load_nlp=True):
        super(ITBDataset, self).__init__(name, dataset_root)

        self.videos = {}
        self.video_scenario={}
        # using json files
        json_path = os.path.join(dataset_root, name+'.json')
        assert os.path.isfile(json_path),'{:} does not exist! Please check that!'.format(json_path)
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
        # load videos
        videos = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)

        for video in videos:
            # print(video, dataset_root, meta_data[video]['video_dir'], meta_data[video]['img_names'])
            video_dir = meta_data[video]['video_dir']
            tmp = meta_data[video]['img_names']
            for i in range(len(tmp)):
                tmp[i] = os.path.join(video_dir, tmp[i])
            meta_data[video]['img_names'] = tmp
            if load_nlp:
                self.videos[video] = ITBVideo(video,
                                              dataset_root,
                                              meta_data[video]['video_dir'],
                                              meta_data[video]['init_rect'],
                                              meta_data[video]['img_names'],
                                              meta_data[video]['gt_rect'],
                                              None,  # meta_data[video]['scenario'],
                                              load_img,
                                              meta_data[video]['phrase'])
            else:
                self.videos[video] = ITBVideo(video,
                                                 dataset_root,
                                                 meta_data[video]['video_dir'],
                                                 meta_data[video]['init_rect'],
                                                 meta_data[video]['img_names'],
                                                 meta_data[video]['gt_rect'],
                                                 None, #meta_data[video]['scenario'],
                                                 load_img)
            self.video_scenario[video]=meta_data[video]['scenario_name']

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())



