import json
import os
import shutil

from tqdm import tqdm

from .dataset import Dataset
from .video import Video

class GOT10kVideo(Video):
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
        super(GOT10kVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)
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
            traj_file = os.path.join(path, name, self.name, self.name+'_001.txt')
            # print(traj_file)
            # if not os.path.exists(traj_file):
                # if self.name == 'FleetFace':
                #     txt_name = 'fleetface.txt'
                # elif self.name == 'Jogging-1':
                #     txt_name = 'jogging_1.txt'
                # elif self.name == 'Jogging-2':
                #     txt_name = 'jogging_2.txt'
                # elif self.name == 'Skating2-1':
                #     txt_name = 'skating2_1.txt'
                # elif self.name == 'Skating2-2':
                #     txt_name = 'skating2_2.txt'
                # elif self.name == 'FaceOcc1':
                #     txt_name = 'faceocc1.txt'
                # elif self.name == 'FaceOcc2':
                #     txt_name = 'faceocc2.txt'
                # elif self.name == 'Human4-2':
                #     txt_name = 'human4_2.txt'
                # else:
                #     txt_name = self.name[0].lower()+self.name[1:]+'.txt'
                # traj_file = os.path.join(path, name, txt_name)
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    if len(pred_traj) != len(self.gt_traj):
                        print(name, len(pred_traj), len(self.gt_traj), self.name)
                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:

    #     self.tracker_names = list(self.pred_trajs.keys())

class GOT10kDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False, load_nlp=True):
        super(GOT10kDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'-mixattr.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            gt_bboxes = []
            with open(os.path.join('H:/Datasets/GOT-10k/TSMtrack', video, video + '_001.txt'), 'r') as f:
                bboxes = f.readlines()
                for bbox in bboxes:
                    bbox = bbox.split(',')
                    bbox = [float(cont) for cont in bbox]
                    gt_bboxes.append(bbox)
            # print(len(gt_bboxes), gt_bboxes)
            meta_data[video]['gt_rect'] = gt_bboxes

            if load_nlp:
                self.videos[video] = GOT10kVideo(video,
                                              dataset_root,
                                              meta_data[video]['video_dir'],
                                              meta_data[video]['init_rect'],
                                              meta_data[video]['img_names'],
                                              meta_data[video]['gt_rect'],
                                              None,
                                              load_img,
                                              meta_data[video]['phrase'])
            else:
                self.videos[video] = GOT10kVideo(video,
                                                 dataset_root,
                                                 meta_data[video]['video_dir'],
                                                 meta_data[video]['init_rect'],
                                                 meta_data[video]['img_names'],
                                                 meta_data[video]['gt_rect'],
                                                 None)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())

if __name__ == '__main__':
    path = 'I:/PycharmProjects/SiamCAR-SPOS/tools/results/GOT-10k/neckexpand-14-0.7_0.06_0.1'
    save_path = 'I:/PycharmProjects/SiamCAR-SPOS-NL/tools/results/GOT-10k/snlt'
    path = 'I:/PycharmProjects/snlt-main/experiments/got10k/EXPR'
    leader_path = 'D:/PycharmProjects/TransT-main/pysot_toolkit/results/GOT-10k/SMT_submission_2021_08_25_02_19_26/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results = os.listdir(path)
    for result in results:
        name = result.split('.')[0]
        if not os.path.exists(os.path.join(save_path, name)):
            os.makedirs(os.path.join(save_path, name))
        # shutil.copyfile(os.path.join(path, result), os.path.join(save_path, name, name+'_001.txt'))
        shutil.copyfile(os.path.join(leader_path, name, name+'_time.txt'), os.path.join(save_path, name, name+'_time.txt'))
        with open(os.path.join(path, result), 'r', encoding='utf-8') as f:
            bboxes = f.readlines()
        with open(os.path.join(leader_path, name, name+'_001.txt'), 'r', encoding='utf-8') as f:
            gt_bbox = f.readlines()[0]
        with open(os.path.join(save_path, name, name+'_001.txt'), 'w', encoding='utf-8') as f:
            # bboxes = f.readlines()
            for i in range(len(bboxes)+1):
                if i == 0:
                    f.write(gt_bbox)
                else:
                    f.write(bboxes[i-1])

