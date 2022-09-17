import os
import json
import numpy as np
import shutil

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


class LaSOTEXTVideo(Video):
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
        super(LaSOTEXTVideo, self).__init__(name, root, video_dir,
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
                    contents = f.readlines()
                    if '\t' in contents[1]:
                        pred_traj = [list(map(float, x.strip().split('\t')))
                                     for x in contents]
                    else:
                        pred_traj = [list(map(float, x.strip().split(',')))
                                for x in contents]
            else:
                print("File not exists: ", traj_file)
            if self.name == 'monkey-17':
                pred_traj = pred_traj[:len(self.gt_traj)]
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj
        self.tracker_names = list(self.pred_trajs.keys())



class LaSOTEXTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, load_nlp=True):
        super(LaSOTEXTDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)

            # for i in range(len(meta_data[video]['img_names'])):
            #     root_name = meta_data[video]['img_names'][i].split('/')[0].split('-')[0]
            #     meta_data[video]['img_names'][i] = root_name + '/' + meta_data[video]['img_names'][i]
            if load_nlp:
                self.videos[video] = LaSOTEXTVideo(video,
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
                self.videos[video] = LaSOTEXTVideo(video,
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

if __name__ == '__main__':
    json_file = dict()

    path = 'J:/Datasets/LaSOT_Extension'
    classes = os.listdir(path)
    print(classes)
    for class_ in classes:
        if class_.endswith('.json'):
            continue
        test_folders = os.listdir(os.path.join(path, class_))

        for test_folder in test_folders:
            # if test_folder != 'Cartoon-WenZi-video-02':
            #     continue
            json_file[test_folder] = dict()
            json_file[test_folder]['video_dir'] = class_+'/'+test_folder

            with open(os.path.join(path, class_, test_folder, 'groundtruth.txt'), 'r', encoding='utf-8') as f:
                contents = f.readlines()
                gt_rect = []
                for content in contents:
                    content = content.strip()
                    if content == '':
                        continue
                    rect = [int(one) for one in content.split(',')]
                    gt_rect.append(rect)
            json_file[test_folder]['gt_rect'] = gt_rect
            json_file[test_folder]['init_rect'] = gt_rect[0]
            print(test_folder)
            # with open(os.path.join(path, test_folder, 'attributes.txt'), 'r', encoding='utf-8') as f:
            #     contents = f.readlines()
            #     contents = [i for i in contents if i != '\n']
            #     attr = [int(content.strip()) for content in contents]
            #     json_file[test_folder]['attr'] = attr
            json_file[test_folder]['attr'] = 'tracking'

            img_files = sorted(os.listdir(os.path.join(path, class_, test_folder, 'img')))
            # print(img_files)
            img_names = []
            for img_file in img_files:
                if not img_file.endswith('.jpg') and not img_file.endswith('.png'):
                    continue
                img_names.append(os.path.join(class_+'/'+test_folder, 'img', img_file).replace('\\', '/'))
            absent = [1 for i in range(len(img_files))]
            json_file[test_folder]['absent'] = absent
            json_file[test_folder]['img_names'] = img_names
            with open(os.path.join(path, class_+'/'+test_folder, 'nlp.txt'), 'r') as f:
                nlp = f.readlines()
                for j in range(len(nlp)):
                    nlp[j] = nlp[j].strip()
                json_file[test_folder]['phrase'] = nlp
            # break
    # print(json_file['Cartoon-WenZi-video-02'])
    jsondata = json.dumps(json_file, indent=4, separators=(',', ': '))
    with open('J:/Datasets/LaSOT_Extension/lasotext.json', 'w', encoding='utf-8') as f:
        f.write(jsondata)


    # path = '/datassd2/TPE_results-CoMo-LTM2-VOT2019/zp_tune'
    # folders = os.listdir(path)
    # for folder in folders:
    #     if 'lr=0.42' in folder:
    #         print(folder)

    # save_path = 'J:/CoMo-LTM-results/newTNL2K'
    # path = 'J:/CoMo-LTM-results/TNL2K'
    # folders = os.listdir(path)
    # for folder in folders:
    #     if not os.path.exists(os.path.join(save_path, folder)):
    #         os.makedirs(os.path.join(save_path, folder))
    #     results = os.listdir(os.path.join(path, folder))
    #     for result in results:
    #         splits = result.split(' ')
    #         if '&' in splits:
    #             splits.remove('&')
    #         new_str = ''.join(splits)
    #         if ' ' in result:
    #             print(result, new_str)
    #         shutil.copyfile(os.path.join(path, folder, result), os.path.join(save_path, folder, new_str))


