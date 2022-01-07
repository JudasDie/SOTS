import _init_paths
import os
import yaml
import time
import numpy
import argparse
from os.path import exists
import utils.read_file as reader
from evaluator.sot_eval import eval_sot

def parse_args():
    """
    configs for onekey to run.
    """
    parser = argparse.ArgumentParser(description='Train with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='experiments/AutoMatch.yaml', help='yaml configure file name')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # parser configs
    info = reader.load_yaml(args.cfg)
    commonINFO, modelINFO, trainINFO, testINFO, tuneINFO = info['COMMON'], info['MODEL'], info['TRAIN'],\
                                                           info['TEST'], info['TUNE']

    DDP = trainINFO['DDP']['ISTRUE']
    model_name = modelINFO['NAME']
    log_dir = commonINFO['LOG_DIR']
    if not exists(log_dir):
        os.makedirs(log_dir)

    # epoch training -- train 50 or more epochs
    if trainINFO['ISTRUE']:
        print('[*] ====> model training <====')
        print('[*] Train Start Time: {}'.format(time.strftime("%Y-%m-%d_%H:%M:%S")))

        # train_script_name = 'train_{}.py'.format(model_name)
        train_log_name = '{0}_train.{1}.log'.format(model_name, time.strftime("%Y-%m-%d_%H:%M:%S"))

        print('python ./tracking/train_sot.py --cfg {0} --wandb 2>&1 | tee logs/{1}'
                  .format(args.cfg, train_log_name))

        if not DDP:
            os.system('python ./tracking/train_sot.py --cfg {0} --wandb | tee logs/{1}'
                    .format(args.cfg, train_log_name))
        else:
            print('====> use DDP for training <====')
            gpu_nums = len([int(i) for i in commonINFO.GPUS.split(',')])
            os.system('python -m torch.distributed.launch --nproc_per_node={0} ./tracking/train_sot.py --cfg {1} --wandb | tee logs/{2}'
                      .format(gpu_nums, args.cfg, train_log_name))

        print('[*] Train End Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))

    # epoch testing -- test 20-50 epochs (or others)
    if testINFO['ISTRUE']:
        print('[*] ====> model testing <====')
        print('[*] Test Start Time: {}'.format(time.strftime("%Y-%m-%d_%H:%M:%S")))

        test_log_name = '{0}_epoch_test.{1}.log'.format(model_name, time.strftime("%Y-%m-%d_%H:%M:%S"))

        print('mpiexec -n {0} python ./tracking/test_epochs.py --cfg {1} --threads {0} 2>&1 | tee logs/{2}'
              .format(testINFO['THREADS'], args.cfg, test_log_name))

        os.system('mpiexec -n {0} python ./tracking/test_epochs.py --cfg {1} --threads {0} 2>&1 | tee logs/{2}'
                  .format(testINFO['THREADS'], args.cfg, test_log_name))

        # test on vot or otb benchmark
        print('[*] ====> evaluation <====')
        trackers = os.listdir(os.path.join('./result', testINFO['DATA']))
        trackers = " ".join(trackers)
        if 'VOT' in testINFO['DATA']:
            print('python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset {0} --tracker_result_dir result/{0} --trackers {1}'.format(testINFO['DATA'], trackers))
            os.system('python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset {0} --tracker_result_dir result/{0} --trackers {1} 2>&1 | tee logs/{2}'.format(testINFO['DATA'], trackers, eval_log_name))
        else:
            evaler = eval_sot()
            evaler.run(dataset=testINFO['DATA'], result_path='./result/', tracker_reg=model_name+'*')

        print('[*] Test End Time: {}'.format(time.strftime("%Y-%m-%d_%H:%M:%S")))

    # tuning -- with TPE
    if tuneINFO['ISTRUE']:
        print('[*] ====> hyper-parameters tuning <====')
        print('[*] Tune Start Time: {}'.format(time.strftime("%Y-%m-%d_%H:%M:%S")))
        tune_log_name = '{0}_tune.{1}.log'.format(model_name, time.strftime("%Y-%m-%d_%H:%M:%S"))

        print('python -u ./tracking/tune_sot.py --cfg {0} 2>&1 | tee logs/{1}'.format(args.cfg, tune_log_name))

        if not exists('logs'):
            os.makedirs('logs')
        os.system('python -u ./tracking/tune_sot.py --cfg {0} 2>&1 | tee logs/{1}'.format(args.cfg, tune_log_name))


if __name__ == '__main__':
    main()
