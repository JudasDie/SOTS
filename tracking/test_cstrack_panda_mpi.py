import _init_paths

#import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.cstrack_panda import JDETracker_panda
from mot_online.log import logger
from mot_online.timer import Timer
from mot_online.evaluation import Evaluator

from mot_online.utils import mkdir_if_missing
from dataset.cstrack_panda import LoadImages_jde, LoadImages_gt, LoadImages_panda
#from opts import opts
from mpi4py import MPI
import sys


def write_results(filename, results, data_type):
    num = 0
    id = []
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
                num += 1
                if track_id not in id:
                    id += [track_id]
    logger.info('save results to {}'.format(filename))
    return num,len(id)


def eval_seq(opt, dataloader, data_type, result_filename,seq,save_dir=None, show_image=True, frame_rate=30, label_dict = {}):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker_panda(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    seq_num = 0
    once = 1
    vertical_list = {}
    for path, img_list, start_list, split_size_list,boundary_list,index_sum_list, img0_list, img0, frame_id_start in dataloader:
    #for path, img, img0, frame_id_start, dense_mask in dataloader:
        if once == 1:
            frame_id = frame_id_start
            once = 0
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = []
        for b_i in range(len(img_list)):
            blob.append(torch.from_numpy(img_list[b_i]).cuda().unsqueeze(0))

        if int(frame_id)+1 in label_dict.keys():
            online_targets = tracker.update(im_blob=blob, img0_list=img0_list, start_list=start_list, split_size_list=split_size_list,
                                            boundary_list=boundary_list,label_list = label_dict[int(frame_id)+1],index_sum_list=index_sum_list,
                                            img0=img0, seq_num=seq, save_dir=save_dir)
        else:
            online_targets = tracker.update(im_blob=blob, img0_list=img0_list, start_list=start_list,
                                            split_size_list=split_size_list,
                                            boundary_list=boundary_list, label_list=[],
                                            index_sum_list=index_sum_list,
                                            img0=img0, seq_num=seq, save_dir=save_dir)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > opt.min_shape_ratio
            vertical_add = round(tlwh[2] / tlwh[3],1)
            if vertical_add not in vertical_list:
                vertical_list[vertical_add] = 0
            vertical_list[vertical_add] += 1
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #if show_image or save_dir is not None:
        #    online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
        #                                  fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        #if save_dir is not None:
        #    cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    result_detection,id_num = write_results(result_filename, results, data_type)
    print(vertical_list)
    return frame_id, timer.average_time, timer.calls,result_detection,id_num,frame_id_start


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    logger.setLevel(logging.INFO)
    result_root = os.path.join('..', 'results')
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    result_detection = []
    result_id = []
    for seq in seqs:
        output_dir = os.path.join('../vis')
        logger.info('start seq: {}'.format(seq))
        yolov5_det_path = os.path.join(opt.det_results,seq.split("/")[-1]+".txt") # get the other detection results
        label_dict = {}
        if os.path.exists(yolov5_det_path):
            with open(yolov5_det_path, "a", encoding="utf-8") as f:
                f = open(yolov5_det_path, "r", encoding="utf-8")
                for line in f:
                    data = line.split(',')
                    if int(data[0]) not in label_dict.keys():
                        label_dict[int(data[0])] = []
                    label_dict[int(data[0])].append([float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])])


        #dataloader = LoadImages_jde(osp.join(data_root, seq), opt.img_size,opt.val_hf)
        #dataloader = LoadImages_panda(osp.join(data_root, seq), opt.img_size, opt.val_hf,
        #                            split_size=[[3840, 2160], [7680, 4320], [15360, 8640]], over_lap=0.3,label_dict=label_dict)
        dataloader = LoadImages_panda(osp.join(data_root, seq), opt.img_size, opt.val_hf,
                                    split_size=[[2560, 1440], [5120, 2880], [10240, 5760]], over_lap=0.3,label_dict=label_dict)
        #dataloader = LoadImages_panda(osp.join(data_root, seq), opt.img_size, opt.val_hf,
        #                            split_size=[], over_lap=0.3,label_dict=label_dict)
        #dataloader = LoadImages_gt(osp.join(data_root, seq, 'img1'), opt.img_size, opt.val_hf)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq.split("/")[-1]))
        frame_rate = 2
        nf, ta, tc, detection_num,id_num,frame_id_start = eval_seq(opt, dataloader, data_type, result_filename,seq,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, label_dict=label_dict)
        result_detection += [detection_num]
        result_id += [id_num]
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename,frame_id_start,nf))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
        sys.stdout.flush()
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    print("detection_num:",result_detection,sum(result_detection))
    print("id_num:", result_id, sum(result_id))
    #Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
    sys.stdout.flush()


if __name__ == '__main__':
    #opt = opts().init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_panda', default=False, help='test panda')
    parser.add_argument('--val_hf', default=False, help='test ours')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou thresh for nms')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--min_box_area', type=float, default=0, help='filter out tiny boxes')
    parser.add_argument('--min_shape_ratio', type=float, default=1.4, help='filter out tiny boxes')
    parser.add_argument('--mean', type=float, default=[0.408, 0.447, 0.470], help='mean for STrack')
    parser.add_argument('--std', type=float, default=[0.289, 0.274, 0.278], help='std for STrack')
    parser.add_argument('--input_video', type=str, default='../videos/MOT16-03.mp4',
                             help='path to the input video')
    parser.add_argument('--output_format', type=str, default='video', help='video or text')
    parser.add_argument('--output_root', type=str, default='/results', help='expected output root path')
    parser.add_argument('--vis_state', type=int, default=0, help='1 for vision or heatmap and reid')
    parser.add_argument('--det_results', type=str, default="../yolov5_panda", help='Other detection results')
    parser.add_argument('--weights',type=str, default='../weights/cstrack_panda.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='../experiments/model_set/CSTrack_panda.yaml', help='model.yaml path')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--fuse_thres', type=float, default=0, help='object confidence threshold')
    parser.add_argument('--data_cfg', type=str,default='../src/lib/cfg/data.json',help='load data from cfg')
    # parser.add_argument('--data_dir', type=str, default='/home/mfx2/tcdata')
    parser.add_argument('--data_dir', type=str, default='../tcdata')
    parser.add_argument('--device', default='0',help='-1 for CPU, use comma for multiple gpus')

    opt = parser.parse_args()
    opt.img_size = (1088, 608)
    opt.gpus = opt.device
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        opt.device = "0"
        if opt.test_panda:
            #seqs_str = '''panda_round2_test_20210331_A_part1/11_Train_Station_Square
            #              '''
            #seqs_str = '''panda_round2_test_20210331_B_part1/14_Ceremony
            #                          '''
            seqs_str = '''panda_round2_train_20210331_part10/10_Huaqiangbei
                                      '''

            data_root = opt.data_dir
        seqs = [seq.strip() for seq in seqs_str.split()]

        main(opt,
             data_root=data_root,
             seqs=seqs,
             exp_name='MOT15_val_all_dla34',
             show_image=False,
             save_images=False,
             save_videos=False)
    elif rank == 1:
        opt.device = "1"
        if opt.test_panda:
            #seqs_str = '''panda_round2_test_20210331_A_part2/12_Nanshan_i_Park
            #              '''
            seqs_str = '''panda_round2_test_20210331_B_part2/15_Dongmen_Street
                                                  '''
            #seqs_str = '''panda_round2_train_20210331_part22/07_University_Campus
            #                          '''
            data_root = opt.data_dir
        seqs = [seq.strip() for seq in seqs_str.split()]

        main(opt,
             data_root=data_root,
             seqs=seqs,
             exp_name='MOT15_val_all_dla34',
             show_image=False,
             save_images=False,
             save_videos=False)
    elif rank == 2:
        opt.device = "2"
        if opt.test_panda:
            seqs_str = '''panda_round2_test_20210331_A_part3/13_University_Playground
                          '''
            data_root = opt.data_dir
        seqs = [seq.strip() for seq in seqs_str.split()]

        main(opt,
             data_root=data_root,
             seqs=seqs,
             exp_name='MOT15_val_all_dla34',
             show_image=False,
             save_images=False,
             save_videos=False)


