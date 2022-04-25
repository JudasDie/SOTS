''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: MOT tracker
Data: 2022.4.7
'''

import os
import cv2
import torch
import numpy as np
from loguru import logger
import torch.nn.functional as F
from collections import deque
from torchsummary import summary

from models.mot_builder import simple_mot_builder
from utils.general_helper import intersect_dicts
from utils.model_helper import check_keys

import tracker.mot_helper.detection_helper as det_helper
import tracker.mot_helper.association_helper as ass_helper
from tracker.mot_helper.motion_helper import KalmanFilter
from tracker.mot_helper.tracker_helper import BaseTrack, TrackState, STrack, joint_stracks, sub_stracks, \
                                              remove_duplicate_stracks, vis_feature


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.model_builder = simple_mot_builder(opt)

        if opt.args.device is not None:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        logger.info('Creating model...')

        ckpt = torch.load(opt.args.weights, map_location=opt.device)  # load checkpoint
        self.model = self.model_builder.build(pre_save_cfg=None).to(opt.device)  # create
        exclude = ['anchor'] if opt.args.cfg else []  # exclude keys
        if type(ckpt['model']).__name__ == "OrderedDict":
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['model'].float().state_dict()  # to FP32

        check_keys(self.model, state_dict)
        state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect

        self.model.load_state_dict(state_dict, strict=False)  # load
        self.model.cuda().eval()
        total_params = sum(p.numel() for p in self.model.parameters())

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def update(self, im_blob, img0,seq_num, save_dir):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        dets = []

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob, augment=False)
            pred, train_out = output[1]

        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        detections = []
        if len(pred) > 0:
            dets,x_inds,y_inds = det_helper.non_max_suppression_and_inds(pred[:,:6].unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres,method='cluster_diou')
            if len(dets) != 0:
                det_helper.scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                id_feature = output[0][0, y_inds, x_inds, :].cpu().numpy()

                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                              (tlbrs, f) in zip(dets[:, :5], id_feature)]
            else:
                detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = ass_helper.embedding_distance(strack_pool, detections)
        #dists = ass_helper.embedding_distance2(strack_pool, detections)
        #dists = ass_helper.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = ass_helper.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = ass_helper.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # vis
        track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = [],[],[],[],[]
        if self.opt.args.vis_state == 1 and self.frame_id % 20 == 0:
            if len(dets) != 0:
                for i in range(0, dets.shape[0]):
                    bbox = dets[i][0:4]
                    cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0, 255, 0), 2)
                track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = ass_helper.vis_id_feature_A_distance(strack_pool, detections)
            vis_feature(self.frame_id, seq_num, img0, track_features,
                                  det_features, cost_matrix, cost_matrix_det, cost_matrix_track, max_num=5, out_path=save_dir)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = ass_helper.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = ass_helper.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = ass_helper.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = ass_helper.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        #logger.debug('===========Frame {}=========='.format(self.frame_id))
        #logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        #logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        #logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        #logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks











