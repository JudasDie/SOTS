''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: read files with [.yaml] [.txt]
Data: 2021.6.23
'''
import os
import time
import math
import copy
import wandb
import logging
from pathlib import Path
import subprocess
import pdb
_cached_git_status = None


def simple_logger(name='root'):
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

logger = simple_logger('root')


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

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def setup_wandb(config, notes):
    '''
    setup wandb online watching
    https://wandb.ai/
    '''

    tags = config.MODEL.NAME if hasattr(config.MODEL, 'NAME') else config.MODEL.Name
    mode = 'online' if config.TRAIN.WANDB_ONLINE else 'offline'

    output_dir = config.COMMON.CHECKPOINT_DIR if hasattr(config.COMMON, 'CHECKPOINT_DIR')  else config.args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    group = None
    project = None
    run_id = None

    # pdb.set_trace()
    project = tags
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    run_id = '{}_{}.log'.format(tags, time_str)

    # network_config = copy.deepcopy(config)
    config.TRAIN.git_version = get_git_status()

    DDP = config.TRAIN.DDP if hasattr(config.TRAIN, 'DDP') else config.args.local_rank !=-1
    RANK = config.TRAIN.DDP.RANK if hasattr(config.TRAIN, 'DDP') else config.args.local_rank
    local_world_size = config.TRAIN.DDP.local_world_size if hasattr(config.TRAIN, 'DDP') else config.world_size

    if DDP:  # TODO: check here for DDP
        group = run_id
        run_id = run_id + f'-rank{RANK // local_world_size}.{RANK}'

    if run_id is not None:
        if len(run_id) > 128:
            run_id = run_id[:128]
            print('warning: run id truncated for wandb limitation')

    wandb_instance = wandb.init(project=project, tags=tags, config=config, force=True, job_type='train', id=run_id, mode=mode, dir=output_dir, group=group, notes=str(notes))
    return wandb_instance


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


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.



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
        elif config.TEST.DATA in ['OTB2013', 'OTB2015', 'LASOTTEST', 'LASOT', 'LASOTEXT']:
            for x in boxes:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif config.TEST.DATA in ['GOT10KVAL', 'GOT10KTEST', 'TNL2K', 'TREK', 'TOTB', 'TRACKINGNET']:
            for x in boxes:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        else:
            for x in boxes:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')


    fin.close()
    if config.TEST.DATA == 'GOT10KTEST' and time_path is not None:
        with open(time_path, "w") as fin:
            for x in times:
                fin.write(str(x) + '\n')

        fin.close()


def mot_benchmark_save(filename, results, data_type):
    """
    save mot evaluation results
    :param filename:
    :param results:
    :param data_type:
    :return:
    """
    num = 0
    id = []
    if data_type == 'MOTChallenge':
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
    return num, len(id)


def _get_git_status():
    """
    check git version (code from SwinTrack)
    """
    global _cached_git_status
    if _cached_git_status is not None:
        return _cached_git_status
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('utf-8').strip()
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'dirty' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    git_state = sha, diff, branch
    _cached_git_status = git_state
    return git_state


def get_git_status_message():
    """
    check git version (code from SwinTrack)
    """
    git_status = _get_git_status()
    git_diff = git_status[1]
    if git_diff == 'dirty':
        git_diff = 'has uncommited changes'
    message = f'sha: {git_status[0]}, diff: {git_diff}, branch: {git_status[2]}'
    return message


def get_git_status():
    """
    check git version (code from SwinTrack)
    """
    return '-'.join(_get_git_status())