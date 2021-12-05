from collections import deque
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary

from core.mot.general import non_max_suppression_and_inds, non_max_suppression_jde, non_max_suppression
from core.mot.torch_utils import intersect_dicts
from models.mot.cstrack import Model,SiamMot

from mot_online import matching
from mot_online.kalman_filter import KalmanFilter
from mot_online.log import logger
from mot_online.utils import *
from mot_online.decode import mot_decode
from mot_online.basetrack import BaseTrack, TrackState
import thop

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, x_i, y_i, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.x_i = x_i
        self.y_i = y_i
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.1


    def update_features(self, feat,save_state = False,update=False):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if save_state == False and update == False:
            if self.smooth_feat is None:
                self.smooth_feat = feat
            else:
                #self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
                self.smooth_feat = self.smooth_feat + self.alpha * (feat - self.smooth_feat)
                self.features.append(feat)
        if save_state == True:
            self.features.append(feat)
        if update == True:
            #self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat = self.smooth_feat + self.alpha * (feat - self.smooth_feat)
            self.features[-1] = feat

        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, save_state=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat, save_state=save_state)
        self.x_i =new_track.x_i
        self.y_i = new_track.y_i
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True, save_state=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.x_i =new_track.x_i
        self.y_i = new_track.y_i
        if update_feature:
            self.update_features(new_track.curr_feat, save_state=save_state)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if int(opt.gpus[0]) >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        ckpt = torch.load(opt.weights, map_location=opt.device)  # load checkpoint
        model_base = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=1).to(opt.device)  # create
        self.model = SiamMot(self.opt,model_base)  # create
        exclude = ['anchor'] if opt.cfg else []  # exclude keys
        if type(ckpt['model']).__name__ == "OrderedDict":
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
        self.model.load_state_dict(state_dict, strict=False)  # load
        self.model.cuda().eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')

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

    def update(self, im_blob, img0,seq_num, save_dir,public_det=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        dets = []
        dets_ori = []
        dets_add = []

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            Tracklets_temp = []
            ori_h,ori_w,_ = img0.shape
            for track in self.tracked_stracks:
                if track.is_activated:
                    if len(track.features) >= 1:
                        Tracklets_temp.append(track.smooth_feat)
            if len(Tracklets_temp) == 0:
                output = self.model(im_blob)
            else:
                out = self.model(im_blob,Tracklets_T=[torch.Tensor(Tracklets_temp).cuda()])
                output, hmmap, siambox = out[0], out[1], out[2]

            pred, train_out = output[2]
            dense_mask = output[1]
            if len(Tracklets_temp) != 0:
                y = siambox
                y[..., 0:2] = ((y[..., 0:2].sigmoid() - 0.5) * train_out[1] + train_out[5][0].to(hmmap.device)) * train_out[4][0]  # xy
                y[..., 2:4] = y[..., 2:4] * 8
                p_siam,xs_siam,ys_siam = mot_decode(hmmap, y, K=100)
                p_siam_index = p_siam[:, :, 4] > self.opt.conf_thres_siammot
                p_siam = p_siam[p_siam_index]
                xs_siam = xs_siam[p_siam_index].squeeze(-1)
                ys_siam = ys_siam[p_siam_index].squeeze(-1)



        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        if len(pred) > 0:
            dets, x_inds, y_inds = non_max_suppression_and_inds(pred[:, :6].unsqueeze(0), self.opt.conf_thres,
                                                                self.opt.nms_thres, dense_mask=dense_mask,
                                                                method='cluster_diou')

        if len(dets) > 0:
            if len(Tracklets_temp) != 0:
                p_siam_n = p_siam.clone()
                p_siam_n[..., :2] = p_siam[..., :2] - p_siam[..., 2:4] / 2
                p_siam_n[..., 2:4] = p_siam[..., :2] + p_siam[..., 2:4] / 2
                p_siam_n = p_siam_n.cpu()
                ious_fuse = np.max(matching.ious(dets[...,:4],p_siam_n[...,:4]),axis=0)
                ious_fuse_index = ious_fuse <= self.opt.iousfuse_thres
                x_inds = x_inds + xs_siam[ious_fuse_index].tolist()
                y_inds = y_inds + ys_siam[ious_fuse_index].tolist()
                if self.opt.vis_state:
                    dets_ori = dets.clone()
                    dets_add = p_siam_n[ious_fuse_index].clone()
                    scale_coords(self.opt.img_size, dets_ori[:, :4], img0.shape).round()
                    scale_coords(self.opt.img_size, dets_add[:, :4], img0.shape).round()
                dets = torch.cat([dets,p_siam_n[ious_fuse_index]],dim=0)

        if len(dets) != 0:
            id_feature = output[0][0, y_inds, x_inds, :].cpu().numpy()
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, x_i, y_i, 30) for
                          (tlbrs, f, x_i, y_i) in zip(dets[:, :5], id_feature, x_inds, y_inds)]
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
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, save_state=True)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, save_state=True)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, save_state=True)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, save_state=True)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
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


        for track in self.tracked_stracks:
            if track.is_activated:
                track.update_features(track.curr_feat,update=True)

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

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
