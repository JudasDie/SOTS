''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: read files with [.yaml] [.txt]
Data: 2021.6.23
'''
import os
import time
import math
import logging
from pathlib import Path


def create_logger(cfg, modelFlag='OCEAN', phase='train'):
    '''
    creat log file for training
    '''

    root_output_dir = Path(cfg.COMMON.LOG_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    model = cfg.MODEL.NAME

    final_output_dir = root_output_dir / model

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = root_output_dir / model / (model + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def print_speed(i, i_time, n, logger):
    '''
    print training speed of each iteration and remaining time
    '''
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))
    logger.info('\nPROGRESS: {:.2f}%\n'.format(100 * i / n))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def sot_benchmark_save_path(config, args, video_info):
    """
    build benchmark save path for VOT tracking
    """
    if args.resume is None or args.resume == 'None':
        resume = config.TEST.RESUME
    else:
        resume = args.resume

    dataset = config.TEST.DATA

    if config.TEST.EPOCH_TEST:
        suffix = resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('result', dataset, config.MODEL.NAME + suffix)
    else:
        tracker_path = os.path.join('result', dataset, config.MODEL.NAME)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video_info['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video_info['name'] + '_001.txt')
        time_path = None
    elif 'GOT10K' in dataset:
        video_path = os.path.join(tracker_path, video_info['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, '{:s}_001.txt'.format(video_info['name']))
        time_path = os.path.join(video_path, '{:s}_time.txt'.format(video_info['name']))
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video_info['name']))
        time_path = None

    return result_path, time_path


def sot_benchmark_save(inputs):
    """
    save sot tracking results to txt files
    """
    boxes, times, result_path, time_path, args, config = inputs['boxes'], inputs['times'], inputs['result_path'], \
                                                 inputs['time_path'], inputs['args'], inputs['config']

    with open(result_path, "w") as fin:
        if 'VOT' in config.TEST.DATA:
            for x in boxes:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        elif config.TEST.DATA in ['OTB2013', 'OTB2015', 'LASOTTEST', 'LASOT']:
            for x in boxes:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif config.TEST.DATA in ['GOT10KVAL', 'GOT10KTEST', 'TNL2K', 'TREK', 'TOTB', 'TRACKINGNET']:
            for x in boxes:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    fin.close()
    if config.TEST.DATA == 'GOT10KTEST' and time_path is not None:
        with open(time_path, "w") as fin:
            for x in times:
                fin.write(str(x) + '\n')

        fin.close()
