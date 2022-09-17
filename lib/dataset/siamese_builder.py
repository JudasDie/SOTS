''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: build Siamese dataset loader
Data: 2021.6.23
'''

from __future__ import division

import os
import cv2
import json
import math
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import utils.box_helper as boxhelper
import utils.augmentation as auger

import sys
sys.path.append('../')

from transformers import BertTokenizer, BertModel

sample_random = random.Random()

class SiameseDataset(Dataset):
    def __init__(self, cfg):
        super(SiameseDataset, self).__init__()
        # pair information
        self.template_size = cfg.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.TRAIN.SEARCH_SIZE
        self.stride = cfg.TRAIN.STRIDE
        self.cfg = cfg

        if cfg.MODEL.NAME in ['Ocean', 'CNNInMo', 'VLT_SCAR']:
            self.score_size = 25
        elif cfg.MODEL.NAME in ['AutoMatch']:
            self.score_size = 31
        elif cfg.MODEL.NAME in ['SiamFC', 'SiamDW']:
            self.score_size = 17
        elif cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            self.score_size = 32
            self.GT_xyxy = False
        else:
            raise Exception('Not implemented model!')

        self.nasnlp = False
        if cfg.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.nasnlp = True

        # data augmentation
        self.color = cfg.TRAIN.DATASET.AUG.COMMON.COLOR
        self.flip = cfg.TRAIN.DATASET.AUG.COMMON.FLIP
        self.rotation = cfg.TRAIN.DATASET.AUG.COMMON.ROTATION
        self.blur = cfg.TRAIN.DATASET.AUG.COMMON.BLUR
        self.gray = cfg.TRAIN.DATASET.AUG.COMMON.GRAY
        self.label_smooth = cfg.TRAIN.DATASET.AUG.COMMON.LABELSMOOTH
        self.mixup = cfg.TRAIN.DATASET.AUG.COMMON.MIXUP
        self.neg = cfg.TRAIN.DATASET.AUG.COMMON.NEG
        self.jitter = None

        self.shift_s = cfg.TRAIN.DATASET.AUG.SEARCH.SHIFTs
        self.scale_s = cfg.TRAIN.DATASET.AUG.SEARCH.SCALEs
        self.shift_e = cfg.TRAIN.DATASET.AUG.EXEMPLAR.SHIFT
        self.scale_e = cfg.TRAIN.DATASET.AUG.EXEMPLAR.SCALE

        # grids on input image
        self.grids()

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
            + ([transforms.Grayscale(num_output_channels=3), ] if self.gray > random.random() else [])
        )

        # train data information
        print('train datas: {}'.format(cfg.TRAIN.DATASET.WHICH_USE))
        self.train_datas = []  # all train dataset
        start = 0
        self.num = 0
        for data_name in cfg.TRAIN.DATASET.WHICH_USE:
            dataset = subData(cfg, data_name, start)
            self.train_datas.append(dataset)
            start += dataset.num  # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        if random.random() < self.neg:  # neg
            template = dataset._get_negative_target(index)
            search = np.random.choice(self.train_datas)._get_negative_target()
            neg = True
        else:
            template, search = dataset._get_pairs(index, dataset.data_name)
            neg = False

        phrase = None
        if self.nasnlp:
            phrase = dataset.get_phrase(index)

        template, search = self.check_exists(index, dataset, template, search)

        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, bbox_t, dag_param_t = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size, search=True)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)

        if self.cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            cls_label = np.array([0])
            cls_label = torch.tensor(cls_label)
        elif self.cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
            cls_label = np.zeros((self.score_size, self.score_size), dtype=np.int64)
        else:
            if neg:
                cls_label = np.zeros((self.score_size, self.score_size))
            else:
                cls_label = self._dynamic_label([self.score_size, self.score_size], dag_param.shift)

        if self.cfg.MODEL.NAME in ['Ocean', 'AutoMatch']:
            reg_label, reg_weight = self.reg_label(bbox)
        else:
            reg_label, reg_weight = None, None

        if self.cfg.MODEL.NAME in ['OceanPlus', 'AutoMatch']:
            template_mask = self.te_mask(bbox_t)
            jitterBox, jitter_ious = self.jitter_box(bbox)
        else:
            template_mask, jitterBox, jitter_ious = None, None, None

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        if self.nasnlp:
            nlp_len = 50
            if phrase == [] or phrase is None:
                # print(phrase, path)
                phrase_ids = torch.zeros(nlp_len, dtype=torch.long)
                phrase_attnmask = torch.zeros(nlp_len, dtype=torch.long)
            else:
                if isinstance(phrase, str):
                    phrase = [phrase]
                phrase = self.bert_tokenizer.batch_encode_plus(phrase, padding='longest', return_tensors='pt')
                phrase_ids = phrase['input_ids'].squeeze()
                phrase_ids = torch.cat([phrase_ids, torch.zeros(nlp_len - phrase_ids.size(0), dtype=torch.long)], dim=0)
                phrase_attnmask = phrase['attention_mask'].squeeze()
                phrase_attnmask = torch.cat([phrase_attnmask, torch.zeros(nlp_len - phrase_attnmask.size(0), dtype=torch.long)], dim=0)


        outputs = {'template': template, 'search': search, 'cls_label': cls_label, 'reg_label': reg_label,
                   'reg_weight': reg_weight, 'template_bbox': np.array(bbox_t, np.float32),
                   'search_bbox': np.array(bbox, np.float32), 'template_mask': template_mask, 'jitterBox': jitterBox,
                   'jitter_ious': jitter_ious, 'phrase_ids': phrase_ids, 'phrase_attnmask': phrase_attnmask}

        outputs = self.data_package(outputs)

        return outputs

    def data_package(self, outputs):
        clean = []
        for k, v in outputs.items():
            if v is None:
                clean.append(k)
        if len(clean) == 0:
            return outputs
        else:
            for k in clean:
                del outputs[k]

        return outputs

    def check_exists(self, index, dataset, template, search):
        name = dataset.data_name
        while True:
            if not (os.path.exists(template[0]) and os.path.exists(search[0])):
                index = random.randint(0, 100)
                template, search = dataset._get_pairs(index, name)
                continue
            else:
                return template, search

    def _shuffle(self):
        """
        random shuffel
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = boxhelper.center2corner(boxhelper.Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """
        draw_image = np.array(image.copy())
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
                    (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)
        cv2.imwrite(name, draw_image)

    def _draw_reg(self, image, grid_x, grid_y, reg_label, reg_weight, save_path, index):
        """
        visiualization
        reg_label: [l, t, r, b]
        """
        draw_image = image.copy()
        # count = 0
        save_name = join(save_path, '{:06d}.jpg'.format(index))
        h, w = reg_weight.shape
        for i in range(h):
            for j in range(w):
                if not reg_weight[i, j] > 0:
                    continue
                else:
                    x1 = int(grid_x[i, j] - reg_label[i, j, 0])
                    y1 = int(grid_y[i, j] - reg_label[i, j, 1])
                    x2 = int(grid_x[i, j] + reg_label[i, j, 2])
                    y2 = int(grid_y[i, j] + reg_label[i, j, 3])

                    draw_image = cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))

        cv2.imwrite(save_name, draw_image)

    def _mixupRandom(self):
        """
        gaussian random -- 0.3~0.7
        """
        return random.random() * 0.4 + 0.3

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size, search=False):
        """
        data augmentation for input pairs
        """
        shape = image.shape
        crop_bbox = boxhelper.center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if search:
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_s), (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            param.shift = (self._posNegRandom() * self.shift_e, self._posNegRandom() * self.shift_e)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_e), (1.0 + self._posNegRandom() * self.scale_e))  # scale change

        crop_bbox, _ = auger.aug_apply(boxhelper.Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = boxhelper.BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = boxhelper.Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)  # other data augmentation
        return image, bbox, param

    def _mixupShift(self, image, size):
        """
        random shift mixed-up image
        """
        shape = image.shape
        crop_bbox = boxhelper.center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * 64, self._posNegRandom() * 64)  # shift
        crop_bbox, _ = boxhelper.aug_apply(boxhelper.Corner(*crop_bbox), param, shape)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image

    # ------------------------------------
    # function for creating training label
    # ------------------------------------
    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        sz_y = sz // 2 + int(-c_shift[1] / 8)

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                         np.ones_like(y),
                         np.where(dist_to_center < rNeg,
                                  0.5 * np.ones_like(y),
                                  np.zeros_like(y)))
        return label

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

    def reg_label(self, bbox):
        """
        generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1  # [17, 17]
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)  # [17, 17, 4]
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def te_mask(self, bbox):
        """
        generate mask for template frame
        :param bbox: [x1, y1, x2, y2]
        :return: binary mask
        """
        x1, y1, x2, y2 = bbox
        mask = np.zeros((self.template_size, self.template_size))
        r_start, r_end = int(y1), math.ceil(y2 + 1)
        c_start, c_end = int(x1), math.ceil(x2 + 1)

        mask[r_start:r_end, c_start:c_end] = 1

        return mask

    def IOUgroup(self, boxes, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy.reshape(4, )
        pred_x1, pred_y1, pred_x2, pred_y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        ww[ww < 0] = 0
        hh[hh < 0] = 0

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (np.abs(area + target_a - inter) + 1)

        overlap[overlap > 1] = 0

        return overlap

    def jitter_box(self, box):
        """
        :param box: [x1, y1, x2, y2] 1*4
        :return:
        """

        box = np.array([box.x1, box.y1, box.x2, box.y2]).reshape(1, 4)
        box_rep = box.repeat(96, axis=0)

        add = np.array([4, 8, 12, 16]).astype(np.float)
        minus = -1 * add
        add2 = add.reshape(4, 1).repeat(2, axis=-1)
        minus2 = minus.reshape(4, 1).repeat(2, axis=1)

        if self.jitter is None:
            shift = np.zeros((96, 4))

            # settle (x1, y1) change (x2, y2)
            shift[0:4, 2] += add
            shift[4:8, 2] += minus
            shift[8:12, 3] += add
            shift[12:16, 3] += minus
            shift[16:20, 2:4] += add2
            shift[20:24, 2:4] += minus2

            # settle (x2, y1) change (x1, y2)
            shift[24:28, 0] += add
            shift[28:32, 0] += minus
            shift[32:36, 3] += add
            shift[36:40, 3] += minus
            shift[40:44, 0] += add
            shift[40:44, 3] += add
            shift[44:48, 0] += minus
            shift[44:48, 3] += minus

            # settle (x2, y2) change (x1, y1)
            shift[48:52, 0] += add
            shift[52:56, 0] += minus
            shift[56:60, 1] += add
            shift[60:64, 1] += minus
            shift[64:68, 0:2] += add2
            shift[68:72, 0:2] += minus2

            # settle (x1, y2) change (x2, y1)
            shift[72:76, 2] += add
            shift[76:80, 2] += minus
            shift[80:84, 1] += add
            shift[84:88, 1] += minus
            shift[88:92, 1:3] += add2
            shift[92:96, 1:3] += minus2

            self.jitter = shift

        jitter_box = box_rep + self.jitter
        jitter_box = np.clip(jitter_box, 0, 255)

        # ious:
        ious = self.IOUgroup(jitter_box, box)

        return jitter_box, ious


# ---------------------
# for a single dataset
# ---------------------
class subData(object):
    """
    for training with multi dataset
    """

    def __init__(self, cfg, data_name, start):
        self.data_name = data_name
        self.start = start

        info = cfg.TRAIN.DATASET.CONFIG[data_name]
        self.frame_range = info.RANGE
        self.num_use = info.USE
        self.root = info.PATH

        with open(info.ANNOTATION) as fin:
            meta_data = json.load(fin)
            self.labels = self._filter_zero(meta_data)
            self._clean()
            self.num = len(self.labels)  # video numer

        self.num_use = self.num if self.num_use == -1 else self.num_use
        self._shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            phrase = None
            for trk, frames in tracks.items():
                if trk == 'phrase':
                    phrase = frames
                    continue
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
                meta_data_new[video]['phrase'] = phrase
        return meta_data_new

    def _clean(self):
        """
        remove empty videos/frames/annos in dataset
        """
        # no frames
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                if track == 'phrase':
                    continue
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))

        for video, track in to_del:
            try:
                del self.labels[video][track]
            except:
                pass

        # no track/annos
        to_del = []

        if self.data_name == 'YTB':
            to_del.append('train/1/YyE0clBPamU')  # This video has no bounding box.
        print(self.data_name)

        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_image_anno(self, video, track, frame):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)

        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def _get_pairs(self, index, data_name):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        video_keys = list(video.keys())
        video_keys.remove('phrase')
        track = random.choice(video_keys)
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames) - 1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]

        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _get_negative_target(self, index=-1):
        """
        dasiam neg
        """
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        video_keys = list(video.keys())
        video_keys.remove('phrase')
        track = random.choice(video_keys)
        track_info = video[track]

        frames = track_info['frames']
        frame = random.choice(frames)

        return self._get_image_anno(video_name, track, frame)

    def get_phrase(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        if 'phrase' not in video.keys():
            return None
        else:
            return video['phrase']


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import utils.read_file as reader
    config = reader.load_yaml('experiment/Ocean.yaml')

    train_set = SiameseDataset(config)
    train_loader = DataLoader(train_set, batch_size=16, num_workers=1, pin_memory=False)

    for iter, input in enumerate(train_loader):
        # label_cls = input[2].numpy()  # BCE need float
        template = input[0]
        search = input[1]
        print(template.size())
        print(search.size())

        print('dataset test')

    print()

