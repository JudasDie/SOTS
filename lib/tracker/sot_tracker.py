''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: build sot tracker
Data: 2021.6.23
'''
import os
import math
import cv2
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
import utils.read_file as reader
import utils.tracking_helper as tracking_helper
from pprint import pprint
import pdb

class SiamTracker(object):
    def __init__(self, config):
        super(SiamTracker, self).__init__()
        self.config = config
        self.dataset = self.config.TEST.DATA

    def init(self, inputs, hp=None):
        """
        initilaize the Siamese tracking networks
        """

        # parse inputs
        im, self.target_pos, self.target_sz, self.model = inputs['image'], inputs['pos'], inputs['sz'], inputs['model']
        p = DefaultConfig()
        self.im_h = im.shape[0]
        self.im_w = im.shape[1]
        p.update({'MODEL_NAME': self.config.MODEL.NAME})
        p.update({'exemplar_size': self.config.TRAIN.TEMPLATE_SIZE})
        p.update({'instance_size': self.config.TRAIN.SEARCH_SIZE})
        p.renew()
        
        # hyperparameters
        if not hp and self.dataset in self.config.TEST.HYPERS.keys():

            cfg_benchmark = self.config.TEST.HYPERS[self.dataset]
            p.update(cfg_benchmark)
            p.renew()

            if 'small_sz' in cfg_benchmark.keys():
                if ((self.target_sz[0] * self.target_sz[1]) / float(self.im_h * self.im_w)) < 0.004:
                    p.instance_size = cfg_benchmark['big_sz']
                    p.renew()
                else:
                    p.instance_size = cfg_benchmark['small_sz']
                    p.renew()

        # for tune
        if hp:
            p.update(hp)
            p.renew()

            # for small object (from DaSiamRPN released)
            if 'big_sz' in hp:
                if ((self.target_sz[0] * self.target_sz[1]) / float(self.im_h * self.im_w)) < 0.004:
                    p.instance_size = hp['big_sz']
                    p.renew()
                else:
                    p.instance_size = hp['small_sz']
                    p.renew()

        self.p = p

        if self.config.MODEL.NAME in ['Ocean', 'OceanPlus', 'AutoMatch']:
            self.window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
            self.grids(p)
        elif self.config.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            hanning = np.hanning(p.score_size)
            window = np.outer(hanning, hanning)
            self.window = window.flatten()
        elif self.config.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
            hanning = np.hanning(p.score_size)
            self.window = np.outer(hanning, hanning)
        else:
            self.window = np.outer(np.hanning(int(p.score_size) * int(p.response_up_FC)),
                              np.hanning(int(p.score_size) * int(p.response_up_FC)))
            self.window /= self.window.sum()

        # crop image for Siamese
        wc_z = self.target_sz[0] + self.p.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.p.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        self.avg_chans = np.mean(im, axis=(0, 1))

        crop_input = {'image': im, 'pos': self.target_pos, 'model_sz': self.p.exemplar_size, 'original_sz': s_z, 'avg_chans': self.avg_chans}
        z_crop_meta = tracking_helper.siam_crop(crop_input, pysot_crop=self.config.MODEL.NAME in ['TransInMo','CNNInMo', 'VLT_SCAR', 'VLT_TT'])
        z_crop, z_crop_info = z_crop_meta['image_tensor'], z_crop_meta['meta_info']

        if self.config.MODEL.NAME in ['AutoMatch']:
            self.jitter = tracking_helper.jitter_shift()
            mask = tracking_helper.generate_psedou_mask(self.target_pos, self.target_sz, (self.im_h, self.im_w))

            crop_input['image'] = mask
            mask_crop_meta = tracking_helper.siam_crop(crop_input, mode='numpy')
            mask_crop = (mask_crop_meta['image_tensor'] > 0.5).astype(np.uint8)
            mask_crop = torch.from_numpy(mask_crop)
            target_box = tracking_helper.get_bbox(s_z, self.p, self.target_sz)

            self.model.template({'template': z_crop.unsqueeze(0).cuda(), 'template_mask': mask_crop.unsqueeze(0).cuda(),
                                 'target_box': target_box})
        else:
            if self.config.MODEL.NAME in ['TransInMo', 'VLT_TT']:
                scale_z = self.p.exemplar_size / s_z
                self.scale_z = scale_z
                z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
                self.inplace = False
                z_crop = tvisf.normalize(z_crop, self.mean, self.std, self.inplace)
            if self.config.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
                input_model = {'template': z_crop.unsqueeze(0).cuda()}
                input_model['nas_list_z'] = inputs['nas_list_z']
                input_model['nas_list_x'] = inputs['nas_list_x']
                input_model['nas_list_nlp'] = inputs['nas_list_nlp']
                target_box = tracking_helper.get_bbox(s_z, self.p, self.target_sz)

                if isinstance(target_box, tuple):
                    template_bbox = torch.tensor(
                        [target_box[0], target_box[1], target_box[2], target_box[3]]).cuda().float().unsqueeze(0)
                else:
                    template_bbox = torch.tensor(
                        [target_box.x1, target_box.y1, target_box.x2, target_box.y2]).cuda().float().unsqueeze(0)
                input_model['template_bbox'] = template_bbox
                input_model['phrase'] = inputs['phrase']
                self.model.template(input_model)
            else:
                self.model.template({'template': z_crop.unsqueeze(0).cuda()})

        # for SiamFC: additional parameters
        if self.config.MODEL.NAME in ['SiamDW', 'SiamFC']:
            scale_z = self.p.exemplar_size / s_z
            d_search = (self.p.instance_size - self.p.exemplar_size) / 2
            pad = d_search / scale_z
            s_x_FC = s_z + 2 * pad
            min_s_x_FC = 0.2 * s_x_FC
            max_s_x_FC = 5 * s_x_FC

            s_x_serise = {'s_x': s_x_FC, 'min_s_x': min_s_x_FC, 'max_s_x': max_s_x_FC}
            self.p.update(s_x_serise)
            self.p.scales = self.p.scale_step_FC ** (range(self.p.num_scale_FC) - np.ceil(self.p.num_scale_FC // 2))


    def track(self, im, gt=None):
        if self.config.MODEL.NAME in ['SiamFC', 'SiamDW']:
            scaled_instance = self.p.s_x * self.p.scales
            scaled_target = [[self.target_sz[0] * self.p.scales], [self.target_sz[1] * self.p.scales]]
            crop_input = {'image': im, 'pos': self.target_pos, 'scaled_instance': scaled_instance,
                          'instance_size': self.p.instance_size, 'avg_chans': self.avg_chans}
            x_crop_meta = tracking_helper.siamfc_pyramid_crop(crop_input)
        else:
            # crop image in subsequent frames
            if self.config.MODEL.NAME in ['TransInMo', 'VLT_TT']:
                hc_x = self.target_sz[1] + (4 - 1) * self.p.context_amount * sum(self.target_sz)
                wc_x = self.target_sz[0] + (4 - 1) * self.p.context_amount * sum(self.target_sz)
                s_x = math.ceil(math.sqrt(wc_x * hc_x))
            else:
                hc_z = self.target_sz[1] + self.p.context_amount * sum(self.target_sz)
                wc_z = self.target_sz[0] + self.p.context_amount * sum(self.target_sz)
                s_z = np.sqrt(wc_z * hc_z)
                scale_z = self.p.exemplar_size / s_z

                if self.config.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
                    self.scale_z = scale_z
                    s_x = s_z * (self.p.instance_size / self.p.exemplar_size)
                else:
                    d_search = (self.p.instance_size - self.p.exemplar_size) / 2  # slightly different from rpn++
                    pad = d_search / scale_z
                    s_x = s_z + 2 * pad
                target_sz_incrop = self.target_sz * scale_z
            crop_input = {'image': im, 'pos': self.target_pos, 'model_sz': self.p.instance_size,
                          'original_sz': tracking_helper.python2round(s_x),
                          'avg_chans': self.avg_chans}
            x_crop_meta = tracking_helper.siam_crop(crop_input, pysot_crop=self.config.MODEL.NAME in ['TransInMo','CNNInMo','VLT_SCAR', 'VLT_TT'])


        x_crop, x_crop_info = x_crop_meta['image_tensor'], x_crop_meta['meta_info']
        # tracking and update state
        if self.config.MODEL.NAME in ['SiamFC', 'SiamDW']:
            x_crop = x_crop.cuda()
        else:
            if self.config.MODEL.NAME in ['TransInMo', 'VLT_TT']:
                x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
                x_crop = tvisf.normalize(x_crop, self.mean, self.std, self.inplace)
            x_crop = x_crop.unsqueeze(0).cuda()
        outputs = self.model.track({'search': x_crop})

        if self.config.MODEL.NAME in ['AutoMatch']:
            cls_score_s1, cls_score_s2, bbox_pred = outputs['cls_s1'], outputs['cls_s2'], outputs['reg']
        else:
            cls_score, bbox_pred = outputs['cls'], outputs['reg']

        if self.config.MODEL.NAME in ['Ocean', 'OceanPlus', 'AutoMatch']:
            if self.config.MODEL.NAME in ['AutoMatch']:
                cls_s1 = F.sigmoid(cls_score_s1).squeeze().cpu().data.numpy()
                cls_s2 = F.sigmoid(cls_score_s2).squeeze().cpu().data.numpy()
                cls_score = 0.5 * cls_s1 + 0.5 * cls_s2
            else:
                cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
            bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

            pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
            pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
            pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
            pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

            # size penalty
            s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz_incrop)))  # scale penalty
            r_c = self.change((target_sz_incrop[0] / target_sz_incrop[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

            penalty = np.exp(-(r_c * s_c - 1) * self.p.penalty_k)
            pscore = penalty * cls_score

            # window penalty
            pscore = pscore * (1 - self.p.window_influence) + self.window * self.p.window_influence

            # get max
            r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

            # to real size
            pred_x1 = pred_x1[r_max, c_max]
            pred_y1 = pred_y1[r_max, c_max]
            pred_x2 = pred_x2[r_max, c_max]
            pred_y2 = pred_y2[r_max, c_max]

            if self.config.MODEL.NAME in ['AutoMatch']:  # zoom, a small trick, can be removed with almost no performance decrease
                zoom_pred, zoom_box = self.zoom([pred_x1, pred_y1, pred_x2, pred_y2])
                zoom_max_idx = zoom_pred.argmax()
                zoom_pred_final = zoom_box[zoom_max_idx].cpu().data.numpy()

                [pred_x1_z, pred_y1_z, pred_x2_z, pred_y2_z] = zoom_pred_final
                pred_x1 = 0.5 * (pred_x1 + pred_x1_z)
                pred_y1 = 0.5 * (pred_y1 + pred_y1_z)
                pred_x2 = 0.5 * (pred_x2 + pred_x2_z)
                pred_y2 = 0.5 * (pred_y2 + pred_y2_z)


            pred_xs = (pred_x1 + pred_x2) / 2
            pred_ys = (pred_y1 + pred_y2) / 2
            pred_w = pred_x2 - pred_x1
            pred_h = pred_y2 - pred_y1

            diff_xs = pred_xs - self.p.instance_size // 2
            diff_ys = pred_ys - self.p.instance_size // 2

            diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

            target_sz_inimg = target_sz_incrop / scale_z

            # size learning rate
            lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * self.p.lr

            # size rate
            res_xs = self.target_pos[0] + diff_xs
            res_ys = self.target_pos[1] + diff_ys
            res_w = pred_w * lr + (1 - lr) * target_sz_inimg[0]
            res_h = pred_h * lr + (1 - lr) * target_sz_inimg[1]

            self.target_pos = np.array([res_xs, res_ys])

            if self.config.MODEL.NAME in ['AutoMatch']:
                self.target_sz = target_sz_inimg * (1 - lr) + lr * np.array([res_w, res_h])
            else:
                self.target_sz = np.array([res_w, res_h])

        elif self.config.MODEL.NAME in ['SiamDW']:
            cls_score = cls_score.squeeze().permute(1, 2, 0).cpu().data.numpy()
            up_size = self.p.response_up_FC * cls_score.shape[0]
            response_map_up = cv2.resize(cls_score, (up_size, up_size), interpolation=cv2.INTER_CUBIC)
            temp_max = np.max(response_map_up, axis=(0, 1))
            s_penaltys = np.array([self.p.scale_penalty_FC, 1., self.p.scale_penalty_FC])
            temp_max *= s_penaltys
            best_scale = np.argmax(temp_max)

            response_map = response_map_up[..., best_scale]
            response_map = response_map - response_map.min()
            response_map = response_map / response_map.sum()

            pscore = (1 - self.p.window_influence) * response_map + self.p.window_influence * self.window
            r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
            p_corr = [c_max, r_max]

            disp_instance_final = p_corr - np.ceil(self.p.score_size * self.p.response_up_FC / 2)
            disp_instance_input = disp_instance_final * self.p.total_stride / self.p.response_up_FC
            disp_instance_frame = disp_instance_input * self.p.s_x / self.p.instance_size
            self.target_pos = self.target_pos + disp_instance_frame

            self.p.s_x = max(self.p.min_s_x, min(self.p.max_s_x, (1 - self.p.lr) * self.p.s_x + self.p.lr * scaled_instance[best_scale]))
            self.target_sz = [(1 - self.p.lr) * self.target_sz[0] + self.p.lr * scaled_target[0][0][best_scale],
                         (1 - self.p.lr) * self.target_sz[1] + self.p.lr * scaled_target[1][0][best_scale]]
        elif self.config.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            def _convert_score(score):
                score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
                score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
                return score

            def _convert_bbox(delta):
                delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
                delta = delta.data.cpu().numpy()
                return delta

            score = _convert_score(cls_score)
            pred_bbox = _convert_bbox(bbox_pred)
            penalty_k = self.config.TEST.HYPERS[self.dataset]['penalty_k']
            if penalty_k != 0:
                penalty = self.cal_penalty(pred_bbox * s_x, penalty_lk=penalty_k)
                score = score * penalty

            win_lr = self.config.TEST.HYPERS[self.dataset]['window_influence']
            hp_lr = self.config.TEST.HYPERS[self.dataset]['lr']
            pscore = score * (1-win_lr) + self.window * win_lr

            best_idx = np.argmax(pscore)
            len_r_c = int(math.sqrt(pscore.shape[0]))
            pscore = pscore.reshape(len_r_c,len_r_c)
            r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
            bbox = pred_bbox[:, best_idx]
            bbox = bbox * s_x
            cx = bbox[0] + self.target_pos[0] - s_x / 2
            cy = bbox[1] + self.target_pos[1] - s_x / 2
            width = bbox[2]
            height = bbox[3]

            if penalty_k != 0:
                s_c = self.change(
                    self.sz(width, height) / self.sz(self.target_sz[0] * self.scale_z,
                                                     self.target_sz[1] * self.scale_z))
                r_c = self.change((self.target_sz[0] / self.target_sz[1]) / (width / height))

                penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
                lr = penalty * np.max(pscore) * hp_lr
                width = lr * width + (1 - lr) * self.target_sz[0]
                height = lr * height + (1 - lr) * self.target_sz[1]
            else:
                width = hp_lr * width + (1 - hp_lr) * self.target_sz[0]
                height = hp_lr * height + (1 - hp_lr) * self.target_sz[1]

            # update state
            self.target_pos = np.array([cx, cy])
            self.target_sz = np.array([width, height])
        elif self.config.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
            cls = self._convert_cls(outputs['cls']).squeeze()
            cen = outputs['cen'].data.cpu().numpy()
            cen = (cen - cen.min()) / cen.ptp()
            cen = cen.squeeze()
            lrtbs = outputs['reg'].data.cpu().numpy().squeeze()

            upsize = (self.config.TEST.SCORE_SIZE - 1) * self.config.TEST.STRIDE + 1
            penalty = self.cal_penalty_lrtb(lrtbs, self.config.TEST.HYPERS[self.dataset]['penalty_k'])
            pscore = penalty * cls * cen
            r_max, c_max = self.config.TEST.SCORE_SIZE//2, self.config.TEST.SCORE_SIZE//2
            if self.config.TEST.hanming:
                hp_score = pscore * (1 - self.config.TEST.HYPERS[self.dataset]['window_influence']) + self.window * self.config.TEST.HYPERS[self.dataset]['window_influence']
            else:
                hp_score = pscore

            hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
            p_score_up = cv2.resize(pscore, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
            cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
            lrtbs = np.transpose(lrtbs, (1, 2, 0))
            lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

            scale_score = upsize / self.config.TEST.SCORE_SIZE
            # get center
            max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
            # get w h
            ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / self.scale_z
            ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / self.scale_z

            s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.target_sz[0] * self.scale_z, self.target_sz[1] * self.scale_z))
            r_c = self.change((self.target_sz[0] / self.target_sz[1]) / (ave_w / ave_h))
            penalty = np.exp(-(r_c * s_c - 1) * self.config.TEST.HYPERS[self.dataset]['penalty_k'])
            lr = penalty * cls_up[max_r_up, max_c_up] * self.config.TEST.HYPERS[self.dataset]['lr']
            new_width = lr * ave_w + (1 - lr) * self.target_sz[0]
            new_height = lr * ave_h + (1 - lr) * self.target_sz[1]

            # clip boundary
            # cx = bbox_clip(new_cx, 0, img.shape[1])
            # cy = bbox_clip(new_cy, 0, img.shape[0])
            # width = bbox_clip(new_width, 0, img.shape[1])
            # height = bbox_clip(new_height, 0, img.shape[0])
            self.target_pos = np.array([new_cx, new_cy])
            self.target_sz = np.array([new_width, new_height])

        self.target_pos[0] = max(0, min(self.im_w, self.target_pos[0]))
        self.target_pos[1] = max(0, min(self.im_h, self.target_pos[1]))
        self.target_sz[0] = max(10, min(self.im_w, self.target_sz[0]))
        self.target_sz[1] = max(10, min(self.im_h, self.target_sz[1]))

        return {'pos': self.target_pos, 'sz': self.target_sz, 'score': pscore[r_max, c_max]}


    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2


    def IOUgroup(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy

        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def zoom(self, box):
        box = np.array(box).reshape(1, 4)
        box_rep = box.repeat(96, axis=0)
        jitter_box = box_rep + self.jitter
        jitter_box = np.concatenate((box, jitter_box), axis=0)

        jitter_box = np.clip(jitter_box, 0, self.p.instance_size)
        jitter_box = torch.from_numpy(jitter_box)

        zoom_pred = self.model.zoom(jitter_box)

        return zoom_pred, jitter_box

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def cal_penalty(self, pred_bbox, penalty_lk):
        bboxes_w = pred_bbox[2, :]
        bboxes_h = pred_bbox[3, :]
        s_c = self.change(
            self.sz(bboxes_w, bboxes_h) / self.sz(self.target_sz[0] * self.scale_z, self.target_sz[1] * self.scale_z))
        r_c = self.change((self.target_sz[0] / self.target_sz[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def cal_penalty_lrtb(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.target_sz[0]*self.scale_z, self.target_sz[1]*self.scale_z))
        r_c = self.change((self.target_sz[0] / self.target_sz[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((self.p.instance_size - (self.config.TEST.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([self.p.instance_size, self.p.instance_size]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (self.config.TEST.SCORE_SIZE - 1) * self.config.TEST.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = tracking_helper.bbox_clip(max_r, 0, self.config.TEST.SCORE_SIZE-1)
        max_c = tracking_helper.bbox_clip(max_c, 0, self.config.TEST.SCORE_SIZE-1)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(self.config.TEST.REGION_S * self.p.exemplar_size)
        max_bbox = int(self.config.TEST.REGION_L * self.p.exemplar_size)
        l_region = int(min(max_c_up_hp, tracking_helper.bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, tracking_helper.bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, tracking_helper.bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, tracking_helper.bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        disp = self.accurate_location(max_r_up,max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.target_pos[0]
        new_cy = disp_ori[0] + self.target_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls


class DefaultConfig(object):
    # MODEL_NAME = 'AutoMatch'  # 'Ocean'
    # exemplar_size = 127  # 127
    # instance_size = 255  # 255

    # MODEL_NAME = 'TransInMo'
    # exemplar_size = 128
    # instance_size = 256

    MODEL_NAME = 'CNNInMo'
    exemplar_size = 127
    instance_size = 255

    penalty_k = 0.034
    window_influence = 0.284
    lr = 0.313
    windowing = 'cosine'
    total_stride = 8

    if MODEL_NAME in ['Ocean', 'OceanPlus']:
        score_size = (instance_size - exemplar_size) // total_stride + 1 + 8
    elif MODEL_NAME in ['SiamDW', 'SiamFC', 'CNNInMo', 'VLT_SCAR']:
        score_size = (instance_size - exemplar_size) // total_stride + 1
    elif MODEL_NAME in ['AutoMatch']:
        score_size = (instance_size - exemplar_size) // total_stride + 14
    elif MODEL_NAME in ['TransInMo', 'VLT_TT']:
        score_size = (instance_size - exemplar_size) // total_stride * 2
    else:
        raise Exception('Unknown model!')

    context_amount = 0.5
    ratio = 0.94

    # SiamFC typical hypers
    response_up_FC = 16  # upsampling ratio for SiamFC series
    scale_penalty_FC = 0.9745
    scale_step_FC = 1.089
    num_scale_FC = 3

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        if self.MODEL_NAME in ['Ocean', 'OceanPlus']:
            self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
        elif self.MODEL_NAME in ['SiamDW', 'SiamFC']:
            self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1
        elif self.MODEL_NAME in ['AutoMatch']:
            self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 14 # for ++
        elif self.MODEL_NAME in ['TransInMo', 'VLT_TT']:
            self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride * 2
        elif self.MODEL_NAME in ['CNNInMo', 'VLT_SCAR']:
            self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8
        else:
            raise Exception('Unknown model!')
