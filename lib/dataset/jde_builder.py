''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: Dataset for MOT trackers: JDE, CSTrack, OMC
Data: 2022.4.7
'''

import os
import cv2
import torch
import random
import os.path as osp
import numpy as np
import utils.box_helper as box_helper

from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from loguru import logger
from utils.augmentation import random_perspective, augment_hsv, cutout
from utils.tracking_helper import letterbox, letterbox_jde
from utils.general_helper import torch_distributed_zero_first, get_hash


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def create_dataloader(root, path, imgsz, batch_size, stride, opt, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, state="train"):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    # Set random number seed
    """
    :param root: dataset root, e.g. /datassd/tracking/MOT
    :param path: see lib/dataset/mot_imgs, e.g. lib/dataset/mot_imgs/mot15.train
    :param imgsz: network input size
    :param batch_size: total batch_size
    :param stride: network max stride
    :param opt: hyper-parameters
    :param augment: bool, whether to use hyper-parameter
    :param cache:
    :param pad:
    :param rect: rect train
    :param rank: used for DDP
    :param world_size: used for DDP
    :param workers:
    :param state:
    :return:
    """

    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(root, path, imgsz, batch_size, opt,
                                      augment=augment,  # augment images
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.args.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      rank=rank,
                                      state=state)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if rank != -1 else None

    def _init_fn(seed):
        return np.random.seed(seed)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             worker_init_fn=_init_fn(2020),
                                             shuffle=True if rank == -1 else False,
                                             sampler=train_sampler,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)

    return dataloader, dataset


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, root, paths, img_size=640, batch_size=16, opt=None, augment=False, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1, state="train"):

        self.opt = opt
        try:
            self.img_files_dict = OrderedDict()
            self.label_files_dict = OrderedDict()
            self.tid_num = OrderedDict()
            self.tid_start_index = OrderedDict()

            # get image and label files
            logger.info('load {} images and labels ...'.format(state))
            for ds, path in paths.items():  # ds: dataset name, e.g. mot15
                with open(path, 'r') as file:
                    self.img_files_dict[ds] = file.readlines()
                    self.img_files_dict[ds] = [osp.join(root, x.strip()) for x in self.img_files_dict[ds]]
                    self.img_files_dict[ds] = list(filter(lambda x: len(x) > 0, self.img_files_dict[ds]))

                self.label_files_dict[ds] = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in self.img_files_dict[ds]]

            # get target ids
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
                logger.info('{0} dataset {1}, max target id: {2} (start from 1)'.format(state, ds, max_index))

            last_index = 0

            logger.info('assign different datasets with different target id ...')
            for i, (k, v) in enumerate(self.tid_num.items()):
                self.tid_start_index[k] = last_index
                last_index += v
                logger.info('{0} dataset {1}, start id: {2}, end id: {3}'.format(state, k, self.tid_start_index[k], last_index))

            self.nID = int(last_index + 1)   # number of ids
            self.nds = [len(x) for x in self.img_files_dict.values()]  # total images for each dataset
            self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]  # total frames of top-i datasets
            self.nF = sum(self.nds)  # total frames
            logger.info('=' * 80)
            logger.info('dataset summary')
            logger.info(self.tid_num)
            logger.info('total # identities: {}'.format(self.nID))
            logger.info('start index: {}'.format(self.tid_start_index))
            logger.info('=' * 80)
            self.img_files = []
            self.label_files = []
            for x in self.img_files_dict.values():
                self.img_files += x
            for x in self.label_files_dict.values():
                self.label_files += x
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % (path)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index, each image is in which batch
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size[1] // 2, -img_size[0] // 2]
        self.stride = stride

        logger.info('check whether there is cache files ...')
        logger.info('TODO: this strategy is not optimal for extream large datasets')  # TODO
        if state == "train":
            cache_path = "train.cache"
        if state == "val":
            cache_path = "val.cache"

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

        logger.info('clear and align target ids (especially for multi-datasets) ...')
        key_l = list(self.tid_start_index.keys())

        for key_i in range(1, len(key_l)):
            start_index = self.cds[key_i]
            if key_i != len(key_l)-1:
                end_index = self.cds[key_i+1]
            else:
                end_index = len(labels)
            for l_i in range(start_index, end_index):  # labels[l_i]  (k, 6) k indicates target numbers
                for l_j in range(len(labels[l_i])):    # labels[l_i][l_j]  (1, 6)
                    if labels[l_i][l_j][1] != -1:
                        labels[l_i][l_j][1] += self.tid_start_index[key_l[key_i]]   # change id to align with multi-datasets
                    else:
                        labels[l_i][l_j][1] = 0        # set -1 to 0
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
            logger.info('use rectangular training ...')
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

        # cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = enumerate(self.label_files)

        if rank in [-1, 0]:
            pbar = tqdm(pbar)

        for i, file in pbar:
            l = self.labels[i]  # label
            l = l[:, 1:]   # TODO: only used for 1 class training (remove class id)

            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % l[0]
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:  # default False, since class id has been removed
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        if not osp.exists('./datasubset/images'): os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not osp.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = box_helper.xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty

            if rank in [-1, 0]:
                pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s' % (os.path.dirname(file) + os.sep)
            logger.info(s)
            assert not augment, '%s. Can not train without labels.' % s

        # cache images into memory for faster training (TODO: WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = self.load_image(i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        if self.mosaic:
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < self.opt.TRAIN.DATASET.AUG.mixup:
                img2, labels2 = self.load_mosaic(random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # letterbox
            shape = [self.img_size[1], self.img_size[0]]
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # load labels
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
            img, labels = random_perspective(img, labels,
                                             degrees=self.opt.TRAIN.DATASET.AUG.degrees,
                                             translate=self.opt.TRAIN.DATASET.AUG.translate,
                                             scale=self.opt.TRAIN.DATASET.AUG.scale,
                                             shear=self.opt.TRAIN.DATASET.AUG.shear,
                                             perspective=self.opt.TRAIN.DATASET.AUG.perspective)

            # Augment colorspace
            augment_hsv(img, hgain=self.opt.TRAIN.DATASET.AUG.hsv_h, sgain=self.opt.TRAIN.DATASET.AUG.hsv_s,
                        vgain=self.opt.TRAIN.DATASET.AUG.hsv_v)

            # Apply cutouts
            if random.random() < self.opt.TRAIN.DATASET.AUG.cutout:
                labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = box_helper.xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < self.opt.TRAIN.DATASET.AUG.flipud:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < self.opt.TRAIN.DATASET.AUG.fliplr:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x608x1088
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def load_mosaic(self, index):
        # loads images in a mosaic
        img, (h0, w0), (h, w) = self.load_image(index)
        shape = [self.img_size[1], self.img_size[0]]
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        np.clip(labels[:, 1], 1, w - 1, out=labels[:, 1])  # use with random_affine
        np.clip(labels[:, 3], 1, w - 1, out=labels[:, 3])  # use with random_affine
        np.clip(labels[:, 2], 1, h - 1, out=labels[:, 2])  # use with random_affine
        np.clip(labels[:, 4], 1, h - 1, out=labels[:, 4])  # use with random_affine
        return img, labels

    @staticmethod
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

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            h0, w0 = img.shape[:2]  # orig hw
            r_w = self.img_size[0] / w0  # resize image to img_size
            r_h = self.img_size[1] / h0
            if r_w != 1 or r_h != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r_w < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r_w), int(h0 * r_h)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

    def cache_labels(self, path='labels.cache'):
        """
        save image and labels to cache file
        :param path: cache name and path
        :return:
        """
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                shape = self.exif_size(image)  # image size (width, height)
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

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes