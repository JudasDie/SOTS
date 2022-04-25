''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: MOT tracker helper
Data: 2022.4.7
'''
import cv2
import numpy as np
from collections import OrderedDict, deque
import tracker.mot_helper.association_helper as ass_helper
from tracker.mot_helper.motion_helper import KalmanFilter


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
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

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
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
        if update_feature:
            self.update_features(new_track.curr_feat)

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


def joint_stracks(tlista, tlistb):
    """
    combine two tracklet lists
    :param tlista:
    :param tlistb:
    :return:
    """
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
    """
    remove tracklets with the same target id
    :param tlista:
    :param tlistb:
    :return:
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """
    remove duplicate tracklets
    :param stracksa:
    :param stracksb:
    :return:
    """
    pdist = ass_helper.iou_distance(stracksa, stracksb)
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


def vis_feature(frame_id,seq_num,img,track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track,max_num=5, out_path='/home/XX/'):
    num_zero = ["0000","000","00","0"]
    img = cv2.resize(img, (778, 435))

    if len(det_features) != 0:
        max_f = det_features.max()
        min_f = det_features.min()
        det_features = np.round((det_features - min_f) / (max_f - min_f) * 255)
        det_features = det_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*512
        for d_f in det_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        det_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(det_features_img, (435, 435))
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1)

    if len(cost_matrix_det) != 0 and len(cost_matrix_det[0]) != 0:
        max_f = cost_matrix_det.max()
        min_f = cost_matrix_det.min()
        cost_matrix_det = np.round((cost_matrix_det - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_det)*10
        for c_m in cost_matrix_det:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_det_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_det_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(track_features) != 0:
        max_f = track_features.max()
        min_f = track_features.min()
        track_features = np.round((track_features - min_f) / (max_f - min_f) * 255)
        track_features = track_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*512
        for d_f in track_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        track_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(track_features_img, (435, 435))
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix_track) != 0 and len(cost_matrix_track[0]) != 0:
        max_f = cost_matrix_track.max()
        min_f = cost_matrix_track.min()
        cost_matrix_track = np.round((cost_matrix_track - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_track)*10
        for c_m in cost_matrix_track:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_track_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_track_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix) != 0 and len(cost_matrix[0]) != 0:
        max_f = cost_matrix.max()
        min_f = cost_matrix.min()
        cost_matrix = np.round((cost_matrix - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix[0])*10
        for c_m in cost_matrix:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    dst_path = out_path + "/" + seq_num + "_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.png'
    cv2.imwrite(dst_path, feature_img)