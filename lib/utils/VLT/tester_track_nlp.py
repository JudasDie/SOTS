import torch
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from colorama import Style, Fore

import tracker.sot_tracker as tracker_builder
import utils.box_helper as boxhelper
from utils.VLT.toolkit.datasets import GOT10kDataset
from utils.VLT.toolkit.evaluation import OPEBenchmark
from utils.VLT.toolkit.utils import overlap_ratio, success_overlap, success_error


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, args, train_loader_itr, tracker, dataset):
    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters

    print('clear bn statics....')
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)

    print('train bn with training set (BN sanitize) ....')
    model.train()

    for step in tqdm.tqdm(range(max_train_iters)):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        batchinfo = train_loader_itr.next()
        for param in model.module.parameters():
            device = param.device
            break

        batch_keys = list(batchinfo.keys())
        template = batchinfo['template'].to(device)
        search = batchinfo['search'].to(device)
        cls_label = batchinfo['cls_label'].type(torch.FloatTensor).to(device)

        # Ocean
        reg_label = batchinfo['reg_label'].float().to(device) if 'reg_label' in batch_keys else None
        reg_weight = batchinfo['reg_weight'].float().to(device) if 'reg_weight' in batch_keys else None

        # OceanPlus
        template_mask = batchinfo['template_mask'].to(device) if 'template_mask' in batch_keys else None

        # AUtoMatch
        template_bbox = batchinfo['template_bbox'].to(device) if 'template_bbox' in batch_keys else None
        search_bbox = batchinfo['search_bbox'].to(device) if 'search_bbox' in batch_keys else None
        jitterBox = batchinfo['jitterBox'].float().to(device) if 'jitterBox' in batch_keys else None
        jitter_ious = batchinfo['jitter_ious'].float().to(device) if 'jitter_ious' in batch_keys else None

        model_inputs = {'template': template, 'search': search, 'cls_label': cls_label, 'reg_label': reg_label,
                        'reg_weight': reg_weight, 'template_bbox': template_bbox, 'search_bbox': search_bbox,
                        'template_mask': template_mask, 'jitterBox': jitterBox, 'jitter_ious': jitter_ious,
                        'nas_list_z': cand[0], 'nas_list_x': cand[1], 'nas_list_nlp': cand[-2:],
                        'phrase_ids': batchinfo['phrase_ids'], 'phrase_attnmask': batchinfo['phrase_attnmask']}

        model_loss = model(model_inputs)

    print('starting test....')
    model.eval()

    auc_reverse = 0
    # choice_ids = list(range(len(dataset)))
    # random.shuffle(choice_ids)
    choice_ids = [81, 13, 2, 69, 79, 24, 39, 88, 14, 56, 30, 70, 74, 82, 22, 31, 92, 57, 91, 50, 29, 64, 38, 37, 54, 18, 17, 58, 73, 11, 35, 66, 36, 84, 98, 21, 10, 49, 1, 9, 4, 45, 67, 48, 40, 63, 53, 86, 60, 44, 72, 42, 85, 94, 97, 78, 71, 16, 52, 96, 27, 95, 62, 7, 6, 89, 76, 28, 90, 23, 43, 0, 12, 47, 65, 75, 32, 34, 59, 83, 77, 20, 19, 25, 51, 26, 80, 99, 41, 8, 5, 46, 87, 93, 15, 3, 68, 55, 61, 33]
    choice_ids = choice_ids[:max_test_iters]
    test_frames_pervideo = 11
    benchmark = OPEBenchmark(dataset)
    # video.name
    for v_idx, video in enumerate(dataset):
        if v_idx not in choice_ids:
            continue
        pred_bboxes = []
        for idx, (img, gt_bbox, phrase) in enumerate(video):
            if idx < test_frames_pervideo:
                if idx == 0:
                    cx, cy, w, h = boxhelper.get_axis_aligned_bbox(np.array(gt_bbox))
                    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                    init_inputs = {'image': img, 'pos': target_pos, 'sz': target_sz, 'model': model.module.eval()}
                    init_inputs['phrase'] = phrase
                    init_inputs['nas_list_z'] = cand[0]
                    init_inputs['nas_list_x'] = cand[1]
                    init_inputs['nas_list_nlp'] = cand[-2:]
                    # print(phrase)
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(init_inputs)  # init tracker
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(pred_bbox)
                else:
                    state = tracker.track(img)
                    location = boxhelper.cxy_wh_2_rect(state['pos'], state['sz'])
                    pred_bboxes.append(location)
        success = benchmark.eval_success(v_idx, pred_bboxes, test_frames_pervideo)
        auc_reverse += 1.01 - np.mean(list(success))

    return auc_reverse/max_test_iters




class OPEBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)

    def eval_success(self, v_dix, tracker_traj, n_frame):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        video = self.dataset[v_dix]
        gt_traj = np.array(video.gt_traj)[:n_frame]

        tracker_traj = np.array(tracker_traj)

        success_ret = success_overlap(gt_traj, tracker_traj, n_frame)
        return success_ret

    def eval_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center = self.convert_bb_to_center(gt_traj)
                tracker_center = self.convert_bb_to_center(tracker_traj)
                thresholds = np.arange(0, 51, 1)
                precision_ret_[video.name] = success_error(gt_center, tracker_center,
                        thresholds, n_frame)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[video.name] = success_error(gt_center_norm,
                        tracker_center_norm, thresholds, n_frame)
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret, precision_ret=None,
            norm_precision_ret=None, show_video_level=False, helight_threshold=0.6, tune_eval=False):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                             key=lambda x:x[1],
                             reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^9}|{:^16}|{:^11}|").format(
                "Tracker name", "Success", "Norm Precision", "Precision")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            else:
                precision = 0
            if norm_precision_ret is not None:
                norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()),
                        axis=0)[20]
            else:
                norm_precision = 0
            print(formatter.format(tracker_name, success, norm_precision, precision))

            if tune_eval:
                return success, norm_precision, precision
        print('-'*len(header))

        if show_video_level and len(success_ret) < 10 \
                and precision_ret is not None \
                and len(precision_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f'{Fore.RED}{success_str}{Style.RESET_ALL}|'
                    else:
                        row += success_str+'|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str+'|'
                print(row)
            print('-'*len(header1))

    def save_result_plots(self, success_ret, precision_ret=None,
                    norm_precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                              key=lambda x: int(x[0].split('_')[-1]),
                              reverse=False)
        tracker_names = [x[0] for x in tracker_auc_]
        x_plot = [int(x[0].split('_')[-1]) for x in tracker_auc_]
        y_plot = [tracker_auc[tracker_name] for tracker_name in tracker_names]
        plt.figure()
        plt.plot(x_plot, y_plot, 'o-')

        plt.plot(x_plot, y_plot, '-')
        plt.scatter(x_plot, y_plot, marker='o', c='r')
        for xx, yy in zip(x_plot, y_plot):
            plt.text(xx, yy, '%.3f' % yy, fontdict={'fontsize': 14})
        plt.show()
