import os
import numpy as np
# from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from colorama import Style, Fore

from ..utils import overlap_ratio, success_overlap, success_error

def success_overlap_perframe(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    # mask = np.sum(gt_bb > 0, axis=1) == 4 #TODO check all dataset
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success, iou

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

    def eval_success(self, eval_trackers=None):
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

        success_ret = {}
        for tracker_name in eval_trackers:
            # video_names = []
            # video_scores = []
            success_ret_ = {}
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
                    # print(video.name)
                    # print(video.video_dir)
                    # print(len(tracker_traj), len(gt_traj), len(video.absent))
                    if len(tracker_traj) < len(video.absent):
                        video.absent = video.absent[:len(tracker_traj)]
                        gt_traj = gt_traj[:len(tracker_traj)]
                        # tracker_traj = tracker_traj[:len(tracker_traj)]
                    elif len(tracker_traj) > len(video.absent):
                        # gt_traj = gt_traj[:len(gt_traj)]
                        tracker_traj = tracker_traj[:len(gt_traj)]
                    # else:
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                success_ret_[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
                # print(video.name, np.mean(success_ret_[video.name]))
                # video_names.append(video.name)
                # video_scores.append(np.mean(success_ret_[video.name]))
            success_ret[tracker_name] = success_ret_
            # save_path = 'J:/' + tracker_name+'.txt'
            # if os.path.exists(save_path):
            #     save_path = 'J:/' + tracker_name+'1.txt'
            # with open(save_path, 'w', encoding='utf-8') as f:
            #     for score in video_scores:
            #         f.write(str(score)+'\n')
            # with open('J:/video_names.txt', 'w', encoding='utf-8') as f:
            #     for name in video_names:
            #         f.write(name+'\n')
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
                    if len(tracker_traj) < len(video.absent):
                        video.absent = video.absent[:len(tracker_traj)]
                        gt_traj = gt_traj[:len(tracker_traj)]
                        # tracker_traj = tracker_traj[:len(tracker_traj)]
                    elif len(tracker_traj) > len(video.absent):
                        # gt_traj = gt_traj[:len(gt_traj)]
                        tracker_traj = tracker_traj[:len(gt_traj)]
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
                    if len(tracker_traj) < len(video.absent):
                        video.absent = video.absent[:len(tracker_traj)]
                        gt_traj = gt_traj[:len(tracker_traj)]
                        # tracker_traj = tracker_traj[:len(tracker_traj)]
                    elif len(tracker_traj) > len(video.absent):
                        # gt_traj = gt_traj[:len(gt_traj)]
                        tracker_traj = tracker_traj[:len(gt_traj)]
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
        # x_smooth = np.linspace(np.array(x_plot).min(), np.array(x_plot).max(), 300)
        # y_smooth = make_interp_spline(x_plot, y_plot)(x_smooth)
        # plt.plot(x_smooth, y_smooth, '-')

        plt.plot(x_plot, y_plot, '-')
        plt.scatter(x_plot, y_plot, marker='o', c='r')
        for xx, yy in zip(x_plot, y_plot):
            plt.text(xx, yy, '%.3f' % yy, fontdict={'fontsize': 14})
        # np.savetxt('test.txt', (x_plot, y_plot))
        # print(x_plot, y_plot)
        plt.show()
        # plt.savefig('test.png')

        # tracker_name_len = max((max([len(x) for x in success_ret.keys()]) + 2), 12)
        # header = ("|{:^" + str(tracker_name_len) + "}|{:^9}|{:^16}|{:^11}|").format(
        #     "Tracker name", "Success", "Norm Precision", "Precision")
        # formatter = "|{:^" + str(tracker_name_len) + "}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        # print('-' * len(header))
        # print(header)
        # print('-' * len(header))
        # for tracker_name in tracker_names:
        #     # success = np.mean(list(success_ret[tracker_name].values()))
        #     success = tracker_auc[tracker_name]
        #     if precision_ret is not None:
        #         precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
        #     else:
        #         precision = 0
        #     if norm_precision_ret is not None:
        #         norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()),
        #                                  axis=0)[20]
        #     else:
        #         norm_precision = 0
        #     print(formatter.format(tracker_name, success, norm_precision, precision))
        # print('-' * len(header))

    def eval_mIoU(self, eval_trackers=None):
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

        mIou_ret = {}
        mIou_scen = {}
        video_scenario = self.dataset.video_scenario

        for tracker_name in eval_trackers:
            mIou_ret_ = {}
            mIou_scen_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                    tracker_traj = tracker_traj[:len(gt_traj), :]
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                _, iou = success_overlap_perframe(gt_traj, tracker_traj, n_frame)
                mIou_ret_[video.name] = np.mean(iou[iou > -0.1])
                save_perframe_rs = True
                if save_perframe_rs:
                    per_rs_file = os.path.join(self.dataset.tracker_path, tracker_name, video.name + '_pfiou.txt')
                    if not os.path.exists(per_rs_file):
                        aa = '\n'
                        ious = [str(x) for x in iou]
                        f = open(per_rs_file, 'w')
                        f.write(aa.join(ious))
                        f.close()
                if video_scenario[video.name] in list(mIou_scen_.keys()):
                    mIou_scen_[video_scenario[video.name]] += mIou_ret_[video.name]
                else:
                    mIou_scen_[video_scenario[video.name]] = mIou_ret_[video.name]

            mIou_ret[tracker_name] = mIou_ret_
            mIou_scen[tracker_name] = mIou_scen_

        return mIou_ret, mIou_scen

    def show_result_ITB(self, mIoU_ret, mIou_scen, success_ret=None, precision_ret=None,
                        norm_precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # scenairo names
        scens = ['human-part', 'sport-ball', '3d-object', 'animal', 'uav', 'human-body', 'vehicle', 'sign-logo',
                 'cartoon'];

        # sort tracker
        tracker_mIoU = {}
        for tracker_name in mIoU_ret.keys():
            mIoU = np.mean(list(mIoU_ret[tracker_name].values()))
            tracker_mIoU[tracker_name] = mIoU
            # compute the score of each scenario

        tracker_mIoU_ = sorted(tracker_mIoU.items(),
                               key=lambda x: x[1],
                               reverse=True)
        tracker_names = [x[0] for x in tracker_mIoU_]

        tracker_name_len = max((max([len(x) for x in mIoU_ret.keys()]) + 2), 12)
        header1 = ("|{:^" + str(tracker_name_len) + "}|{:^8}{:^8}{:^9}{:^9}{:^6}{:^8}{:^10}{:^7}{:^9}|{:^13}|").format(
            "Tracker", "human", "sport", "   3D   ", "      ", "   ", "human", "    ", "sign", " ", " overall ")
        header2 = ("|{:^" + str(
            tracker_name_len) + "}|{:^7}|{:^7}|{:^8}|{:^8}|{:^5}|{:^7}|{:^9}|{:^6}|{:^9}|{:^6}|{:^6}|").format(
            "name", "part", "ball", "object", "animal", "uav", "body", "vehicle", "logo", "cartoon", "mIoU", "Suc.")

        # header = ("|{:^" + str(tracker_name_len) + "}|{:^9}|{:^16}|{:^11}|{:^6}|").format(
        #     "Tracker name", "Success", "Norm Precision", "Precision", "mIoU")
        formatter = "|{:^" + str(
            tracker_name_len) + "}|{:^7.1f} {:^7.1f} {:^8.1f} {:^8.1f} {:^5.1f} {:^7.1f} {:^9.1f} {:^6.1f} {:^9.1f}|{:^6.1f}|{:^6.1f}|"
        print('-' * len(header1))
        print(header1)
        print(header2)
        print('-' * len(header1))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            if success_ret is not None:
                success = np.mean(list(success_ret[tracker_name].values()))
            else:
                success = 0
            sce_iou = mIou_scen[tracker_name]
            print(formatter.format(tracker_name, sce_iou[scens[0]] * 5, sce_iou[scens[1]] * 5, sce_iou[scens[2]] * 5,
                                   sce_iou[scens[3]] * 5,
                                   sce_iou[scens[4]] * 5, sce_iou[scens[5]] * 5, sce_iou[scens[6]] * 5,
                                   sce_iou[scens[7]] * 5,
                                   sce_iou[scens[8]] * 5, tracker_mIoU[tracker_name] * 100, success * 100))

        print('-' * len(header1))

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
            print('-' * len(header1))
            print(header1)
            print('-' * len(header1))
            print(header2)
            print('-' * len(header1))
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
                        row += success_str + '|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str + '|'
                print(row)
            print('-' * len(header1))

