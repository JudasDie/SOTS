from collections import deque
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary

from core.mot.general import non_max_suppression_and_inds, non_max_suppression_jde, non_max_suppression
from core.mot.nms_pytorch import soft_nms_pytorch,cluster_nms,cluster_SPM_nms,cluster_diounms,cluster_SPM_dist_nms
from core.mot.torch_utils import intersect_dicts
from models.mot.cstrack import Model

from mot_online import matching_panda
from mot_online.kalman_filter import KalmanFilter
from mot_online.log import logger
from mot_online.utils import *


from mot_online.basetrack import BaseTrack, TrackState
import sys
from mot_online.ensemble_boxes_CSTrack import weighted_boxes_fusion_CSTrack
torch.set_printoptions(precision=8)

def WBF_process(im0, detScale):
    h,w,_ = im0.shape
    prediction_box = detScale.cpu()
    prediction_box[:,0] /= w
    prediction_box[:,1] /= h
    prediction_box[:,2] /= w
    prediction_box[:,3] /= h
    boxes = prediction_box[:,:4].tolist()  # Format and coordinates conversion
    scores = prediction_box[:,4].tolist()
    labels = prediction_box[:,5].tolist()
    return boxes, scores, labels

def WBF_fuse(im0, detScale_list, idf_scale_list, weights=[1,1], iou_thres=0.5, conf_thres=0.5):
    h,w,_ = im0.shape
    boxes_list,scores_list,labels_list,id_feature_list = [],[],[],[]
    for detScale in detScale_list:
        boxes,scores,labels = WBF_process(im0,detScale)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    for idf_scale in idf_scale_list:
        idfeature = idf_scale.cpu().tolist()
        id_feature_list.append(idfeature)
    
    boxes, scores, labels, id_feature = weighted_boxes_fusion_CSTrack(boxes_list, scores_list, labels_list, id_feature_list, weights=weights, iou_thr=iou_thres, skip_box_thr=conf_thres)
    boxes = np.array(boxes)
    scores = np.array(scores)[:,np.newaxis] # increase dimension
    labels = np.array(labels)[:,np.newaxis]
    boxes[:,0] *= w
    boxes[:,1] *= h
    boxes[:,2] *= w
    boxes[:,3] *= h
    output = np.hstack((boxes,scores))
    output = np.hstack((output,labels))
    return torch.from_numpy(output), torch.from_numpy(id_feature)

def fuse_all_det(prediction, conf_thres=0.5, nms_thres=0.4, method='standard',merge=False, dense_mask=""):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate([prediction]):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        #pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        if method == 'standard':
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == 'soft':
            nms_indices = soft_nms_pytorch(pred[:, :4], pred[:, 4], sigma=0.5, thresh=0.2, cuda=1)
        elif method == "cluster":
            nms_indices = cluster_nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == "cluster_SPM":
            nms_indices = cluster_SPM_nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == "cluster_diou":
            nms_indices = cluster_diounms(pred[:, :4], pred[:, 4], nms_thres, dense_mask)
        elif method == "cluster_SPM_dist":
            nms_indices = cluster_SPM_dist_nms(pred[:, :4], pred[:, 4], nms_thres)
        else:
            raise ValueError('Invalid NMS type!')

        if merge and (1 < nP < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(pred[:, :4], pred[:, :4]) > nms_thres  # iou matrix
            weights = iou * pred[:, 4][None]  # box weights
            pred[:, :4] = torch.mm(weights, pred[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            #nms_indices = nms_indices[iou.sum(1) > 1]  # require redundancy

        det_max = pred[nms_indices]
        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    #nms_indices = del_include(output[0][:, :4],output[0][:, 4],0.5)
    #output[0] = output[0][nms_indices]
    return output[0].cpu(), nms_indices

def box_iou_self(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None])  # iou = inter / (area1 + area2 - inter)

def del_more_allinclude(pred, id_feature, iou_threshold: float = 0.5):
    iou = box_iou_self(pred[..., :4], pred[..., :4]).t().cpu().numpy()
    matrix_eye =torch.eye(len(iou))
    matrix_eye = matrix_eye == 1
    iou[matrix_eye] = 0
    ious_keep = np.max(iou, axis=0)
    keep_index = ious_keep < iou_threshold
    pred = pred[keep_index]
    id_feature = id_feature[keep_index]
    return pred,id_feature

def del_more(dets,id_feature,img_size,boundary,big_thres = 0.3):
    w = (dets[:,2] - dets[:,0]) / img_size[0]
    h = (dets[:,3] - dets[:,1]) / img_size[1]
    size_state = (w < big_thres) * (h < big_thres*2)
    x1_state = (0 <= dets[:, 0]) * (dets[:, 0] <= img_size[0])
    x2_state = (0 <= dets[:, 2]) * (dets[:, 2] <= img_size[0])
    y1_state = (0 <= dets[:, 1]) * (dets[:, 1] <= img_size[1])
    y2_state = (0 <= dets[:, 3]) * (dets[:, 3] <= img_size[1])
    boundary_state = [x1_state,x2_state,y1_state,y2_state]
    index = size_state
    for b_i in range(len(boundary)):
        if boundary[b_i] == 0:
            index *= boundary_state[b_i]
    return dets[index],id_feature[index]


###redetection
def area_nms(prediction, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate([prediction]):
        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        conf = torch.ones((len(pred))).float().cuda()
        nms_indices = nms(pred[:, :4], conf, nms_thres)
        det_max = pred[nms_indices]
        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))
    return output[0].cpu()


def split_img(img0, center_list=[]):
    h = len(img0)
    w = len(img0[0])
    img0_list = []
    start_list = center_list[:,:2]
    boundary_list = []
    for location in center_list:
        x1 = int(location[0]) if int(location[0]) > 0 else 0
        x2 = int(location[2]) if int(location[2]) < w else w-1
        y1 = int(location[1]) if int(location[1]) > 0 else 0
        y2 = int(location[3]) if int(location[3]) < h else h-1
        img0_list.append(img0[y1:y2, x1:x2])
        boundary_list.append([0, 0, 0, 0])
    img_list = []
    for i in range(len(img0_list)):
        img, _, _, _ = letterbox_jde(img0_list[i])
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img_list.append(img)

    return img_list, img0_list, start_list, boundary_list

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
        self.alpha = 0.5 #0.5

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
        self.is_activated = True
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


class JDETracker_panda(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if int(opt.gpus[0]) >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')

        ckpt = torch.load(opt.weights, map_location=opt.device)  # load checkpoint
        self.model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=1).to(opt.device)  # create
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
        # self.det_thresh = 0.6
        
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def update(self, im_blob=[], img0_list=[], start_list=[], split_size_list=[], boundary_list=[],label_list=[],index_sum_list=[], img0=[], seq_num="", save_dir=""):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        dets = []

        ''' Step 1: Network forward, get detections & embeddings'''

        det_list = []
        idf_list = []
        yolov5_idf = {}
        det_scale_dict = {}
        idf_scale_dict = {}

        for ib in range(len(im_blob)):
            with torch.no_grad():
                output = self.model(im_blob[ib], augment=False)
                pred, train_out = output[2]
                dense_mask = output[1]
            pred = pred[pred[:, :, 4] > self.opt.conf_thres]
            if len(pred) > 0:
                dets,x_inds,y_inds = non_max_suppression_and_inds(pred[:,:6].unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres,dense_mask=dense_mask,method='cluster_diou')
                # del_small

                if len(dets) != 0:
                    small_thres = 20
                    dets_wh_thres = dets[:, 2:4] - dets[:, :2]
                    det_thres = torch.minimum(dets_wh_thres[:, 0], dets_wh_thres[:, 1])
                    if split_size_list[ib] > 5000:
                        dets = dets[det_thres > small_thres]
                        x_inds = np.array(x_inds)
                        y_inds = np.array(y_inds)
                        x_inds = x_inds[torch.where(det_thres > small_thres)]
                        y_inds = y_inds[torch.where(det_thres > small_thres)]
                        x_inds = x_inds.tolist()
                        y_inds = y_inds.tolist()

                if len(dets) != 0:
                    if not isinstance(y_inds, list): y_inds = [y_inds]
                    if not isinstance(x_inds, list): x_inds = [x_inds]
                    id_feature = output[0][0, y_inds, x_inds, :]
                    dets,id_feature = del_more(dets,id_feature,self.opt.img_size,big_thres = 0.3,boundary=boundary_list[ib])

                if len(dets) != 0:
                    scale_coords(self.opt.img_size, dets[:, :4], img0_list[ib].shape).round()
     
                    dets[..., 0] += start_list[ib][0]
                    dets[..., 2] += start_list[ib][0]
                    dets[..., 1] += start_list[ib][1]
                    dets[..., 3] += start_list[ib][1]
                    if det_list == []:
                        det_list = dets
                        idf_list = id_feature
                    else:
                        det_list = torch.cat((det_list, dets), dim=0)
                        idf_list = torch.cat((idf_list, id_feature), dim=0)
                    if split_size_list[ib] not in det_scale_dict.keys():
                        det_scale_dict[split_size_list[ib]] = dets
                        idf_scale_dict[split_size_list[ib]] = id_feature
                    else:
                        det_scale_dict[split_size_list[ib]] = torch.cat((det_scale_dict[split_size_list[ib]], dets), dim=0)
                        idf_scale_dict[split_size_list[ib]] = torch.cat((idf_scale_dict[split_size_list[ib]], id_feature), dim=0)
        
            if len(index_sum_list[ib]) != 0:
                for yolo_ind in index_sum_list[ib]:
                    yolo_ind_y = yolo_ind[2]
                    yolo_ind_x = yolo_ind[1]
                    if yolo_ind_y < 0: yolo_ind_y = 0
                    if yolo_ind_y > 75: yolo_ind_y = 75
                    if yolo_ind_x < 0: yolo_ind_x = 0
                    if yolo_ind_x > 135: yolo_ind_x = 135
                    yolov5_idf[yolo_ind[0]] = output[0][0, yolo_ind_y, yolo_ind_x, :]

        if len(label_list) != 0:
            label_list = torch.tensor(label_list)
            label_add_list = torch.zeros((len(label_list),1))

            label_list[:, 3] += label_list[:, 1]
            label_list[:, 4] += label_list[:, 2]

            label_list = torch.cat((label_list[:,1:],label_list[:,:1],label_add_list),dim=1)
            yolov5_idf_list = []
            for d_key in sorted(yolov5_idf.keys()):
                if d_key == 0:
                    yolov5_idf_list = yolov5_idf[d_key].unsqueeze(0)
                else:
                    yolov5_idf_list = torch.cat((yolov5_idf_list,yolov5_idf[d_key].unsqueeze(0)),dim=0)

            det_list = torch.cat((det_list, label_list), dim=0)  # concat yolov5 detection results
            idf_list = torch.cat((idf_list, yolov5_idf_list), dim=0)

        # print(torch.mean(det_list[:,:4],dim=0))
        det_list,idf_list = WBF_fuse(img0, [det_list,det_scale_dict[2560]], [idf_list,idf_scale_dict[2560]], weights=[1,1], iou_thres=0.5, conf_thres=0.5)  # WBF fuse boxes and id embeddings with scale of 2560
        id_feature = idf_list.to(self.opt.device) 
        dets = det_list
        '''
        dets,fuse_i = fuse_all_det(det_list[:, :6].cuda(), conf_thres=self.opt.fuse_thres, nms_thres=self.opt.nms_thres,
                                     method='cluster_diou', merge=False,dense_mask=dense_mask)
        id_feature = idf_list[fuse_i]
        '''
        sys.stdout.flush()

        #dets, id_feature = del_more_allinclude(dets, id_feature, iou_threshold = 1)
        if len(dets) != 0:
            id_feature = id_feature.cpu().numpy()
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
        dists = matching_panda.embedding_distance(strack_pool, detections)
        dists = matching_panda.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching_panda.linear_assignment(dists, thresh=0.6)

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
        if self.opt.vis_state == 1:
            if len(dets) != 0:
                for i in range(0, dets.shape[0]):
                    bbox = dets[i][0:4]
                    cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0, 255, 0), 10)
                track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = matching_panda.vis_id_feature_A_distance(strack_pool, detections)
            vis_feature(self.frame_id,seq_num,img0,track_features,
                                  det_features, cost_matrix, cost_matrix_det, cost_matrix_track, max_num=5, out_path=save_dir)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching_panda.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching_panda.linear_assignment(dists, thresh=0.8)#0.5

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
        dists = matching_panda.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching_panda.linear_assignment(dists, thresh=0.7)
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
        if self.frame_id == 1:
            output_stracks = [track for track in self.tracked_stracks if not track.is_activated]
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
    pdist = matching_panda.iou_distance(stracksa, stracksb)
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
    dst_path = out_path + "/" + seq_num.split("/")[-1] + "_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.jpg'
    cv2.imwrite(dst_path, img)
    '''
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
        cutpff_line = [40]*len(cost_matrix[0])*5
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
    '''