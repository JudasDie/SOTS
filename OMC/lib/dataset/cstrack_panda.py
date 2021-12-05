# dataset utils and dataloaders

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import OrderedDict
import os.path as osp
from core.mot.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from core.mot.general import check_requirements, check_file, check_dataset, xywh2xyxy, xywhn2xyxy, xyxy2xywhn, \
    xyn2xy, segments2boxes, clean_str, xyxy2xywh
from core.mot.torch_utils import torch_distributed_zero_first
from core.mot.nms_pytorch import diou

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
num_threads = min(8, os.cpu_count())  # number of multiprocessing threads
logger = logging.getLogger(__name__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def letterbox_jde(img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(root, path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False,
                      pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    setup_seed(2020)
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(root, path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      rank=rank)  # prefix=prefix), image_weights=image_weights

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader

    # loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    def _init_fn(seed):
        return np.random.seed(seed)

    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        worker_init_fn=_init_fn(2020),
                        shuffle=True,
                        sampler=sampler,
                        pin_memory=False,
                        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadImages_jde:  # for inference
    def __init__(self, path, img_size=(1088, 608), val_hf=False):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.frame_id = 0
        if val_hf == 1:
            self.files = self.files[:int(len(self.files) * 0.5) + 1]
        if val_hf == 2:
            self.frame_id = int(len(self.files) * 0.5) + 1
            self.files = self.files[int(len(self.files) * 0.5) + 1:]
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
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox_jde(img0, height=self.height, width=self.width)

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
        img, _, _, _ = letterbox_jde(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadImages_panda:  # for inference
    def __init__(self, path, img_size=(1088, 608), val_hf=False, split_size=[], over_lap=0.3, label_dict={}):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.frame_id = 0
        if val_hf == 1:
            self.files = self.files[:int(len(self.files) * 0.5) + 1]
        if val_hf == 2:
            self.frame_id = int(len(self.files) * 0.5) + 1
            self.files = self.files[int(len(self.files) * 0.5) + 1:]
        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        self.split_size = split_size
        self.over_lap = over_lap
        self.label_dict = label_dict

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        if int(img_path.split("_")[-1].split(".")[0]) in self.label_dict.keys():
            img0_list, start_list, split_size_list, boundary_list, index_sum_list = self.split_img(img0,
                                                                                                   output_size_list=self.split_size,
                                                                                                   over_lap=self.over_lap,
                                                                                                   label_list=
                                                                                                   self.label_dict[int(
                                                                                                       img_path.split(
                                                                                                           "_")[
                                                                                                           -1].split(
                                                                                                           ".")[0])])
        else:
            img0_list, start_list, split_size_list, boundary_list, index_sum_list = self.split_img(img0,
                                                                                                   output_size_list=self.split_size,
                                                                                                   over_lap=self.over_lap,
                                                                                                   label_list=[])

        img_list = []
        for i in range(len(start_list)):
            img, _, _, _ = letterbox_jde(img0_list[i], height=self.height, width=self.width)

            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            img_list.append(img)

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img_list, start_list, split_size_list, boundary_list, index_sum_list, img0_list, img0, self.frame_id

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox_jde(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files

    def split_img(self, img, output_size_list=[], over_lap=0.3, label_list=[]):
        h = len(img)
        w = len(img[0])
        img_list = []
        start_list = []
        split_size_list = []
        boundary_list = []
        index_sum_list = []
        label_sort_list = [0] * len(label_list)
        for output_size in output_size_list:
            num_i = int((w // (output_size[0] - output_size[0] * over_lap)) + 1)
            num_j = int((h // (output_size[1] - output_size[1] * over_lap)) + 1)
            for i in range(num_i):
                w_l = 0
                w_r = 0
                h_up = 0
                h_down = 0
                if i == 0:
                    x_now = 0
                    w_l = 1
                else:
                    x_now += (output_size[0] - output_size[0] * over_lap)
                if x_now > w - output_size[0]:
                    x_now = w - output_size[0]
                    w_r = 1
                for j in range(num_j):
                    if j == 0:
                        y_now = 0
                        h_up = 1
                    else:
                        y_now += (output_size[1] - output_size[1] * over_lap)
                    if y_now > h - output_size[1]:
                        y_now = h - output_size[1]
                        h_down = 1
                    start_list.append([x_now, y_now])
                    img_list.append(img[int(y_now):int(y_now + output_size[1]), int(x_now):int(x_now + output_size[0])])
                    split_size_list.append(output_size[0])
                    boundary_list.append([w_l, w_r, h_up, h_down])
                    index_list = []
                    for label_i in range(len(label_list)):
                        if (x_now < label_list[label_i][1] < x_now + output_size[0] and y_now < label_list[label_i][
                            2] < y_now + output_size[1]) or (
                                x_now < label_list[label_i][1] + label_list[label_i][3] < x_now + output_size[
                            0] and y_now < label_list[label_i][2] + label_list[label_i][4] < y_now + output_size[1]):
                            x_list = sorted([label_list[label_i][1] - x_now,
                                             label_list[label_i][1] + label_list[label_i][3] - x_now, 0,
                                             output_size[0]])
                            y_list = sorted([label_list[label_i][2] - y_now,
                                             label_list[label_i][2] + label_list[label_i][4] - y_now, 0,
                                             output_size[1]])
                            iou = (x_list[2] - x_list[1]) * (y_list[2] - y_list[1]) / (
                                    label_list[label_i][3] * label_list[label_i][4])
                            x_c = (label_list[label_i][1] + label_list[label_i][3] / 2 - x_now) / output_size[0]
                            y_c = (label_list[label_i][2] + label_list[label_i][4] / 2 - y_now) / output_size[1]
                            w_c = label_list[label_i][3] / output_size[0]
                            h_c = label_list[label_i][4] / output_size[1]
                            if output_size == output_size_list[0]:
                                if (w_c <= 0.7 and h_c <= 1) and iou > 0.2:
                                    if label_sort_list[label_i] == 0:
                                        index_list.append([label_i, int(x_c * (self.width // 8)), int(y_c * (self.height // 8))])
                                        label_sort_list[label_i] = 1
                            else:
                                if (0.01 <= w_c <= 0.7 and 0.01 <= h_c <= 1) and iou > 0.2:
                                    if label_sort_list[label_i] == 0:
                                        index_list.append([label_i, int(x_c * (self.width // 8)), int(y_c * (self.height // 8))])
                                        label_sort_list[label_i] = 1
                    index_sum_list.append(index_list)
                    if y_now == h - output_size[1]:
                        break
                if x_now == w - output_size[0]:
                    break

        start_list.append([0, 0])
        img_list.append(img)
        split_size_list.append(w)
        boundary_list.append([1, 1, 1, 1])
        index_list = []
        for label_i in range(len(label_list)):
            if label_sort_list[label_i] == 0:
                x_c = (label_list[label_i][1] + label_list[label_i][3] / 2) / w
                y_c = (label_list[label_i][2] + label_list[label_i][4] / 2) / h
                index_list.append([label_i, int(x_c * (self.width // 8)), int(y_c * (self.height // 8))])
                label_sort_list[label_i] = 1

        index_sum_list.append(index_list)
        return img_list, start_list, split_size_list, boundary_list, index_sum_list


class LoadImages_gt:  # for inference
    def __init__(self, path, img_size=(1088, 608), val_hf=False):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]
        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.files]
        self.down_ratio = 8

        self.frame_id = 0
        if val_hf == 1:
            self.files = self.files[:int(len(self.files) * 0.7) + 1]
        if val_hf == 2:
            self.frame_id = int(len(self.files) * 0.7) + 1
            self.files = self.files[int(len(self.files) * 0.7) + 1:]
        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        self.img_size = img_size
        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]
        label_path = self.label_files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox_jde(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        labels = []
        dense_mask = torch.zeros((self.img_size[1] // 8, self.img_size[0] // 8))
        inds_x = []
        inds_y = []
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels = labels[:, 1:]
            labels[:, 0] = labels[:, 1] * len(img0[0])
            labels[:, 1] = labels[:, 2] * len(img0)
            labels[:, 2] = labels[:, 3] * len(img0[0])
            labels[:, 3] = labels[:, 4] * len(img0)
            labels[:, 4] = 1
            labels_center = labels.copy()
            labels[:, 0] = labels_center[:, 0] - labels_center[:, 2] / 2
            labels[:, 1] = labels_center[:, 1] - labels_center[:, 3] / 2
            labels[:, 2] = labels_center[:, 0] + labels_center[:, 2] / 2
            labels[:, 3] = labels_center[:, 1] + labels_center[:, 3] / 2
            labels_T = torch.from_numpy(labels[:, :4])
            iou = diou(labels_T, labels_T, delta=0.7).triu_(diagonal=1)
            iou = iou + iou.T
            max_iou, _ = torch.max(iou, dim=0)

            labels0 = labels0[:, 2:4]
            output_w = self.width / self.down_ratio
            output_h = self.height / self.down_ratio
            labels0[:, 0] = labels0[:, 0] * output_w
            labels0[:, 1] = labels0[:, 1] * output_h
            for i in range(len(labels0)):
                x = int(labels0[i][0])
                y = int(labels0[i][1])
                inds_x += [x]
                inds_y += [y]
                for i_m in range(int(labels[i][1] * (self.img_size[1] // 8) / len(img0)), int(labels[i][3] * (self.img_size[1] // 8) / len(img0))):
                    for j_m in range(int(labels[i][0] * (self.img_size[0] // 8) / len(img0[0])), int(labels[i][2] * (self.img_size[0] // 8) / len(img0[0]))):
                        if j_m <= 0: j_m = 0
                        if j_m >= output_w: j_m = int(output_w) - 1
                        if i_m < 0: i_m = 0
                        if i_m >= output_h: i_m = int(output_h) - 1
                        dense_mask[i_m][j_m] = max_iou[i] + 0.05

            inds_x = np.zeros((len(labels0),), dtype=np.int64)
            inds_y = np.zeros((len(labels0),), dtype=np.int64)
            for i in range(len(labels0)):
                inds_x[i] = int(labels0[i][0])
                y = int(labels0[i][1])
                inds_x += [x]
                inds_y += [y]
                for i_m in range(int(labels[i][1] * (self.img_size[1] // 8) / len(img0)), int(labels[i][3] * (self.img_size[1] // 8) / len(img0))):
                    for j_m in range(int(labels[i][0] * (self.img_size[0] // 8) / len(img0[0])), int(labels[i][2] * (self.img_size[0] // 8) / len(img0[0]))):
                        if j_m <= 0: j_m = 0
                        if j_m >= output_w: j_m = int(output_w) - 1
                        if i_m < 0: i_m = 0
                        if i_m >= output_h: i_m = int(output_h) - 1
                        dense_mask[i_m][j_m] = max_iou[i] + 0.05
            return img_path, img, img0, self.frame_id, dense_mask

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]
        label_path = self.label_files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox_jde(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        labels = []
        inds_x = []
        inds_y = []
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels = labels[:, 1:]
            labels[:, 0] = labels[:, 1] * len(img0[0])
            labels[:, 1] = labels[:, 2] * len(img0)
            labels[:, 2] = labels[:, 3] * len(img0[0])
            labels[:, 3] = labels[:, 4] * len(img0)
            labels[:, 4] = 1
            labels_center = labels.copy()
            labels[:, 0] = labels_center[:, 0] - labels_center[:, 2] / 2
            labels[:, 1] = labels_center[:, 1] - labels_center[:, 3] / 2
            labels[:, 2] = labels_center[:, 0] + labels_center[:, 2] / 2
            labels[:, 3] = labels_center[:, 1] + labels_center[:, 3] / 2

            labels0 = labels0[:, 2:4]
            output_w = self.width / self.down_ratio
            output_h = self.height / self.down_ratio
            labels0[:, 0] = labels0[:, 0] * output_w
            labels0[:, 1] = labels0[:, 1] * output_h
            inds_x = np.zeros((len(labels0),), dtype=np.int64)
            inds_y = np.zeros((len(labels0),), dtype=np.int64)
            for i in range(len(labels0)):
                inds_x[i] = int(labels0[i][0])
                inds_y[i] = int(labels0[i][1])
                if inds_x[i] < 0: inds_x[i] = 0
                if inds_x[i] >= output_w: inds_x[i] = output_w - 1
                if inds_y[i] < 0: inds_y[i] = 0
                if inds_y[i] >= output_h: inds_y[i] = output_h - 1

        return img_path, img, img0, self.frame_id, labels, [inds_x, inds_y]

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, root, paths, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
                 image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1, state="train"):

        # try:
        self.img_files_dict = OrderedDict()
        self.label_files_dict = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files_dict[ds] = file.readlines()
                self.img_files_dict[ds] = [osp.join(root, x.strip()) for x in self.img_files_dict[ds]]
                self.img_files_dict[ds] = list(filter(lambda x: len(x) > 0, self.img_files_dict[ds]))

            self.label_files_dict[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt').replace('.PNG',
                                                                                                               '.txt')
                # self.label_files_dict[ds] = [x.replace('img1', 'labels_with_ids').replace('.PNG', '.txt').replace(
                # '.jpg', '.txt')
                for x in self.img_files_dict[ds]]

        for ds, label_paths in self.label_files_dict.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files_dict.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)
        self.img_files = []
        self.label_files = []
        for x in self.img_files_dict.values():
            self.img_files += x
        for x in self.label_files_dict.values():
            self.label_files += x
        # except Exception as e:
        #     raise Exception('Error loading data from %s: %s\nSee %s' % (paths, e, help_url))
        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        self.mosaic_border = [-img_size[1] // 2, -img_size[0] // 2]
        self.stride = stride

        # Define labels
        # self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
        #                    self.img_files]

        # Check cache
        # cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if state == "train":
            cache_path = "../train.cache"
        if state == "test":
            cache_path = "../val.cache"
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)
        key_l = list(self.tid_start_index.keys())
        for key_i in range(len(key_l)):
            start_index = self.cds[key_i]
            if key_i != len(key_l) - 1:
                end_index = self.cds[key_i + 1]
            else:
                end_index = len(labels)
            for l_i in range(start_index, end_index):
                for l_j in range(len(self.labels[l_i])):
                    if self.labels[l_i][l_j][1] != -1:
                        self.labels[l_i][l_j][1] += self.tid_start_index[key_l[key_i]]
                    else:
                        self.labels[l_i][l_j][1] = 0
            '''  
                    if labels[l_i][l_j][2] >= 1:
                        labels[l_i][l_j][2] = 0.999
                    if labels[l_i][l_j][2] <= 0:
                        labels[l_i][l_j][2] = 0.001
                    if labels[l_i][l_j][3] >= 1:
                        labels[l_i][l_j][3] = 0.999
                    if labels[l_i][l_j][3] <= 0:
                        labels[l_i][l_j][3] = 0.001
                    '''
        for l_i in range(len(labels)):
            for l_j in range(len(labels[l_i])):
                if labels[l_i][l_j][1] == -1:
                    labels[l_i][l_j][1] = 0

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size[0] / stride + pad).astype(np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = enumerate(self.label_files)
        if rank in [-1, 0]:
            pbar = tqdm(pbar)
        for i, file in pbar:
            l = self.labels[i]  # label
            l = l[:, 1:]
            # l[:,0] = 0
            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % l[0]
                # assert (l >= 0).all(), 'negative labels: %s' % l
                # assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % l
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            if rank in [-1, 0]:
                pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = [None, None]
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            # shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            shape = [self.img_size[1], self.img_size[0]]
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            # if not self.mosaic:
            '''
            img, labels = random_perspective(img, labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])
            '''
            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #    labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            dense_mask = torch.zeros((self.img_size[1] // 8, self.img_size[0] // 8))  # 76, 136
            labels_T = torch.from_numpy(labels[:, 1:5])
            iou = diou(labels_T, labels_T, delta=0.7).triu_(diagonal=1)
            iou = iou + iou.T
            max_iou, _ = torch.max(iou, dim=0)
            for box_i in range(len(labels)):
                y1 = max(int(labels[box_i][2] * (self.img_size[1] // 8) / img.shape[0]), 0)
                y2 = min(int(labels[box_i][4] * (self.img_size[1] // 8) / img.shape[0]), (self.img_size[1] // 8) - 1)
                x1 = max(int(labels[box_i][1] * (self.img_size[0] // 8) / img.shape[1]), 0)
                x2 = min(int(labels[box_i][3] * (self.img_size[0] // 8) / img.shape[1]), (self.img_size[0] // 8) - 1)
                for i_m in range(y1, y2):
                    for j_m in range(x1, x2):
                        dense_mask[i_m][j_m] = max_iou[box_i] + 0.05
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                dense_mask = torch.flipud(dense_mask)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                dense_mask = torch.fliplr(dense_mask)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, dense_mask.unsqueeze(0)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, dense_mask = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, torch.cat(dense_mask, 0)


'''
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None


        try:
            f = []  # image files
            # print(path)
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == 0.4 and cache['hash'] == get_hash(self.label_files + self.img_files)
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(num_threads).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(num_threads) as pool:
            pbar = tqdm(pool.imap_unordered(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {help_url}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = 0.4  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4
'''


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size[0] / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a mosaic
    img, (h0, w0), (h, w) = load_image(self, index)
    # Labels
    shape = [self.img_size[1], self.img_size[0]]
    img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
    shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

    # Load labels
    labels = []
    x = self.labels[index]
    if x.size > 0:
        # Normalized xywh to pixel xyxy format
        labels = x.copy()
        labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
        labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
        labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
    # np.clip(labels[:, 1], 1, w-1, out=labels[:, 1])  # use with random_affine
    # np.clip(labels[:, 3], 1, w-1, out=labels[:, 3])  # use with random_affine
    # np.clip(labels[:, 2], 1, h-1, out=labels[:, 2])  # use with random_affine
    # np.clip(labels[:, 4], 1, h-1, out=labels[:, 4])  # use with random_affine
    return img, labels


"""
def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size[0]
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4
"""


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size[0]
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in img_formats, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                assert f.read() == b'\xff\xd9', 'corrupted JPEG'

        # verify labels
        segments = []  # instance segments
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, 'labels require 5 columns each'
                assert (l >= 0).all(), 'negative labels'
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, ''
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', verbose=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128.zip', verbose=True)
    
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 6) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert os.system(f'unzip -q {path} -d {path.parent}') == 0, f'Error unzipping {path}'
            data_dir = path.with_suffix('')  # dataset directory
            return True, data_dir, list(data_dir.rglob('*.yaml'))[0]  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_file(yaml_path)) as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    nc = data['nc']  # number of classes
    stats = {'nc': nc, 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split], augment=False, rect=True)  # load dataset
        if split == 'train':
            cache_path = Path(dataset.label_files[0]).parent.with_suffix('.cache')  # *.cache path
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=nc))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

    # Save, print and return
    with open(cache_path.with_suffix('.json'), 'w') as f:
        json.dump(stats, f)  # save stats *.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
        # print(yaml.dump([stats], sort_keys=False, default_flow_style=False))
    return stats
