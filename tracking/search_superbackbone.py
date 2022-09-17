import _init_paths
import os
import pdb
import wandb
import json
import torch
import socket
import pprint
import argparse
from contextlib import nullcontext
from easydict import EasyDict as edict

import torch.distributed as dist
import utils.read_file as reader
import utils.log_helper as recorder
import utils.model_helper as loader
import utils.lr_scheduler as learner
import utils.sot_builder as builder

from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.siamese_builder import SiameseDataset as data_builder
from core.trainer.siamese_train import siamese_train as trainer

from utils.VLT.tester_track_nlp import *
from utils.VLT.toolkit.datasets import DatasetFactory

eps = 1e-5

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--cfg', type=str, default='experiments/VLT_TT.yaml', help='yaml configure file name')
parser.add_argument('--wandb', action='store_true', help='use wandb to watch training')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='compulsory for pytorch launcer')
parser.add_argument('--dataset', type=str, default='NASSNLP', help='datasets')
parser.add_argument('--datasetpath', default='path to NASSNLP', type=str, help='datasets path')

parser.add_argument('--resume', default=None, type=str, help='ckpt path')

parser.add_argument('--log-dir', type=str, default='search_cand_logdir')
parser.add_argument('--max-epochs', type=int, default=20)
parser.add_argument('--select-num', type=int, default=10)
parser.add_argument('--population-num', type=int, default=50)
parser.add_argument('--m_prob', type=float, default=0.1)
parser.add_argument('--crossover-num', type=int, default=25)
parser.add_argument('--mutation-num', type=int, default=25)
parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
parser.add_argument('--max-train-iters', type=int, default=50)  # 100
parser.add_argument('--max-test-iters', type=int, default=20)  # 40
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=200)
args = parser.parse_args()


class DataIterator(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


class EvolutionSearcher(object):
    def __init__(self, args, train_loader, dist_model, tracker, dataset):
        self.args = args
        self.train_loader = train_loader

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit

        self.model = dist_model
        self.tracker = tracker
        self.dataset = dataset

        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'nas_checkpoint.pth')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        # self.nr_layer = 20
        self.nr_layer_z = 20
        self.nr_layer_x = 20
        self.nr_layer_nlp = 4
        self.nr_state = 4

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, 'model_spos.pth'))
        jsondata = json.dumps(self.keep_top_k, indent=4, separators=(',', ': '))
        # with open('tnl2k.json', 'w', encoding='utf-8') as f:
        #     f.write(jsondata)
        with open(os.path.join(self.log_dir, 'candidate.txt'), 'w', encoding='utf-8') as f:
            # f.write(self.keep_top_k)
            f.write(jsondata)
        print('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_name)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand[0]) == self.nr_layer_z and len(cand[1]) == self.nr_layer_x
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        # if 'flops' not in info:
        #     info['flops'] = get_cand_flops(cand)
        # print(cand, info['flops'])

        # if info['flops'] > self.flops_limit:
        #     print('flops limit exceed')
        #     return False

        info['err'] = get_cand_err(self.model, cand, self.args, self.train_loader, self.tracker, self.dataset)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: (tuple([np.random.randint(self.nr_state) for i in range(self.nr_layer_z)]), tuple([np.random.randint(self.nr_state) for i in range(self.nr_layer_x)]),
            tuple([np.random.randint(self.nr_state) for i in range(self.nr_layer_nlp)]),
            tuple([np.random.randint(self.nr_state) for i in range(self.nr_layer_nlp)])))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer_z):
                if np.random.random_sample() < m_prob:
                    cand = list(list(s) for s in cand)
                    cand[0][i] = np.random.randint(self.nr_state)
                    cand = tuple([tuple(s) for s in cand])
            for i in range(self.nr_layer_x):
                if np.random.random_sample() < m_prob:
                    cand = list(list(s) for s in cand)
                    cand[1][i] = np.random.randint(self.nr_state)
                    cand = tuple([tuple(s) for s in cand])
            for i in range(self.nr_layer_nlp):
                if np.random.random_sample() < m_prob:
                    cand = list(list(s) for s in cand)
                    cand[2][i] = np.random.randint(self.nr_state)
                    cand = tuple([tuple(s) for s in cand])
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                print(ops)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        self.save_checkpoint()


class DataIterator(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data


def main():
    config = edict(reader.load_yaml(args.cfg))

    if config.TRAIN.DDP.ISTRUE:
        local_rank = config.TRAIN.DDP.LOCAL_RANK if args.local_rank == -1 else args.local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    # create logger
    print('====> create logger <====')
    logger, _, tb_log_dir = recorder.create_logger(config, config.MODEL.NAME, 'train')
    # logger.info(pprint.pformat(config))
    logger.info(config)

    # create tensorboard logger
    print('====> create tensorboard <====')
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # create model
    print('====> build model <====')
    if 'Siam' in config.MODEL.NAME or config.MODEL.NAME in ['Ocean', 'OceanPlus', 'AutoMatch', 'TransT', 'CNNInMo',
                                                            'TransInMo', 'VLT_SCAR', 'VLT_TT']:
        siambuilder = builder.Siamese_builder(config)
        model = siambuilder.build()
    else:
        raise Exception('Not implemented model type!')

    model = model.cuda()
    logger.info(model)
    if config.MODEL.NAME in ['VLT_SCAR', 'VLT_TT']:
        model.backbone.nas(nas_ckpt_path=config.MODEL.NAS_CKPT_PATH)
        model.backbone.load_nlp()
        # model = loader.load_pretrain(model, config.MODEL.BACKBONE.PRETRAIN, f2b=False, addhead=False)  # load pretrain
    elif config.MODEL.NAME in ['SiamDW', 'Ocean', 'OceanPlus', 'AutoMatch']:
        model = loader.load_pretrain(model, './pretrain/{0}'.format(config.TRAIN.PRETRAIN), f2b=True, addhead=True)    # load pretrain

    print('===> init Siamese <====')
    if args.resume is None or args.resume == 'None':
        resume = config.TEST.RESUME
    else:
        resume = args.resume

    if config.MODEL.NAME == 'AutoMatch':
        model = loader.load_pretrain(model, resume, addhead=True, print_unuse=False)
    else:
        model = loader.load_pretrain(model, resume, print_unuse=False)

    # get optimizer
    if not config.TRAIN.START_EPOCH == config.TRAIN.UNFIX_EPOCH and not config.MODEL.NAME in ['SiamDW', 'SiamFC']:
        optimizer, lr_scheduler = learner.build_siamese_opt_lr(config, model, config.TRAIN.START_EPOCH)
    else:
        if config.MODEL.NAME in ['SiamDW', 'SiamFC']:
            trainable_params = loader.check_trainable(model, logger, print=False)
            optimizer, lr_scheduler = learner.build_simple_siamese_opt_lr(config, trainable_params)
        else:
            optimizer, lr_scheduler = learner.build_siamese_opt_lr(config, model, 0)  # resume wrong (last line)

    # check trainable again
    print('==========check trainable parameters==========')
    trainable_params = loader.check_trainable(model, logger)  # print trainable params info

    # create parallel
    gpus = [int(i) for i in config.COMMON.GPUS.split(',')]
    gpu_num = world_size = len(gpus)  # or use world_size = torch.cuda.device_count()
    gpus = list(range(0, gpu_num))

    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    if not config.TRAIN.DDP.ISTRUE:
        device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
        model = DataParallel(model, device_ids=gpus).to(device)
    else:
        local_rank = config.TRAIN.DDP.LOCAL_RANK if args.local_rank == -1 else args.local_rank
        device = torch.device("cuda", local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                        find_unused_parameters=True)

    logger.info(lr_scheduler)
    logger.info('model prepare done')

    train_set = data_builder(config)
    if not config.TRAIN.DDP.ISTRUE:
        train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH * gpu_num,
                                  num_workers=config.TRAIN.WORKERS,
                                  pin_memory=True, sampler=None, drop_last=True)
        train_loader = DataIterator(train_loader)
    else:
        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42)

        train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH, shuffle=False,
                                  num_workers=config.TRAIN.WORKERS, sampler=sampler, pin_memory=True,
                                  drop_last=True)

    dataset_root = os.path.join(args.datasetpath, args.dataset) if args.dataset != 'GOT-10k' else os.path.join(
        args.datasetpath, args.dataset, 'val')
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    tracker = tracker_builder.SiamTracker(config)

    # inputs = {'data_loader': train_loader, 'model': model, 'optimizer': optimizer, 'device': device,
    #           'epoch': epoch + 1, 'cur_lr': curLR, 'config': config,
    #           'writer_dict': writer_dict, 'logger': logger, 'wandb_instance': wandb_instance}
    # model, writer_dict = trainer(inputs)

    # t = time.time()

    searcher = EvolutionSearcher(args, train_loader, model, tracker, dataset)
    searcher.search()

    # print('total searching time = {:.2f} hours'.format(
    #     (time.time() - t) / 3600))


if __name__ == '__main__':
    main()
