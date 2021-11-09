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

from tracker.cstrack import JDETracker
from mot_online.log import logger
from mot_online.timer import Timer
from mot_online.evaluation import Evaluator

from mot_online.utils import mkdir_if_missing
from dataset.cstrack import LoadImages_jde
#from opts import opts


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


def eval_seq(opt, dataloader, data_type, result_filename,seq,save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    seq_num = 0
    once = 1
    vertical_list = {}
    for path, img, img0,frame_id_start in dataloader:
        if once == 1:
            frame_id = frame_id_start
            once = 0
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0, seq,save_dir)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 0.6
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
    logger.setLevel(logging.INFO)
    result_root = os.path.join('..', 'results', exp_name)
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
        dataloader = LoadImages_jde(osp.join(data_root, seq, 'img1'), opt.img_size,opt.val_hf)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc, detection_num,id_num,frame_id_start = eval_seq(opt, dataloader, data_type, result_filename,seq,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
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
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    #opt = opts().init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mot16', default=False, help='test mot16')
    parser.add_argument('--val_mot15', default=False, help='val mot15')
    parser.add_argument('--test_mot15', default=False, help='test mot15')
    parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
    parser.add_argument('--test_mot17', default=False, help='test mot17')
    parser.add_argument('--val_mot17', default=False, help='val mot17')
    parser.add_argument('--val_mot20', default=False, help='val mot20')
    parser.add_argument('--test_mot20', default=False, help='test mot20')
    parser.add_argument('--val_hf', type=int, default=0, help='val_hf')

    parser.add_argument('--nms_thres', type=float, default=0.6, help='iou thresh for nms')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument('--mean', type=float, default=[0.408, 0.447, 0.470], help='mean for STrack')
    parser.add_argument('--std', type=float, default=[0.289, 0.274, 0.278], help='std for STrack')
    parser.add_argument('--input_video', type=str, default='../videos/MOT16-03.mp4',
                             help='path to the input video')
    parser.add_argument('--output_format', type=str, default='video', help='video or text')
    parser.add_argument('--output_root', type=str, default='/results', help='expected output root path')
    parser.add_argument('--vis_state', type=int, default=0, help='1 for vision or heatmap and reid')

    parser.add_argument('--weights',type=str, default='../weights/cstrack.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='../experiments/model_set/CSTrack.yaml', help='model.yaml path')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')

    parser.add_argument('--data_cfg', type=str,default='../src/lib/cfg/data.json',help='load data from cfg')
    parser.add_argument('--data_dir', type=str, default='/data/lc/JDE_data')
    parser.add_argument('--device', default='0',help='-1 for CPU, use comma for multiple gpus')

    opt = parser.parse_args()
    opt.img_size = (1088, 608)
    opt.gpus = opt.device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        seqs_str = '''Venice-2'''
        #seqs_str = '''KITTI-13
        #              KITTI-17
        #              ETH-Bahnhof
        #              ETH-Sunnyday
        #              PETS09-S2L1
        #              TUD-Campus
        #              TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT15_val_all_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
