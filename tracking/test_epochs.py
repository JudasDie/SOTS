''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: epoch test on multiple gpus (prefer 8)
Data: 2021.6.23
'''
import _init_paths
import os
import time
import argparse
from mpi4py import MPI
import utils.read_file as reader
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--cfg', required=True, help='config file')
parser.add_argument('--threads', default=16, type=int, required=True)
args = parser.parse_args()


config = edict(reader.load_yaml(args.cfg))

gpu_nums = len([int(i) for i in config.COMMON.GPUS.split(',')])

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % gpu_nums
node_name = MPI.Get_processor_name()  # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

# run test scripts -- two epochs on each GPU
for i in range(2):
    try:
        epoch_ID += args.threads   # for 16 queue
    except:
        epoch_ID = rank % (config.TEST.END_EPOCH - config.TEST.START_EPOCH + 1) + config.TEST.START_EPOCH

    if epoch_ID > config.TEST.END_EPOCH:
        continue

    resume = 'snapshot/checkpoint_e{}.pth'.format(epoch_ID)
    print('==> test {}th epoch'.format(epoch_ID))

    os.system('python ./tracking/test_sot.py --cfg {0} --resume {1}'.format(args.cfg, resume))
