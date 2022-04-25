''' Details
Author: Zhipeng Zhang/Chao Liang
Function: train MOT methods
Date: 2022.4.7
'''

import _init_paths
import os
import yaml
import time
import math
import json
import wandb
import socket
import torch
import random
import argparse
import numpy as np
import os.path as osp

import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
from contextlib import nullcontext

import utils.log_helper as recorder
import utils.read_file as reader
import utils.git_helper as git_helper
import utils.model_helper as model_helper
from utils.general_helper import setup_seed, increment_dir
from utils.tracking_helper import parser_mot_train_data
from utils.general_helper import select_device, init_seeds, attempt_download, intersect_dicts, \
                                 plot_lr_scheduler, plot_labels, plot_labels, plot_images, jde_fitness, \
                                 torch_distributed_zero_first, strip_optimizer, flush_exname

from models.mot import eval
from models.mot.optimizer import Lookahead
from models.mot_builder import simple_mot_builder
from models.mot.loss import JDELoss, labels_to_class_weights, labels_to_image_weights
from dataset.jde_builder import create_dataloader
import pdb

setup_seed(2020)



def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='experiments/CSTrack.yaml', help='model.yaml path')

    # some advanced training settings
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--multi_scale', action='store_true', help='vary img-size +/- 50%%')

    # task dependent
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')

    # devices and DDP
    parser.add_argument('--device', default='6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of workers for dataloader')

    # results related
    parser.add_argument('--output_dir', type=str, default='snapshot', help='logging and ckpt saving directory')
    parser.add_argument('--name', type=str, default=None, help='unique label for saving logs and ckpts')

    args = parser.parse_args()

    return args


def epoch_train(opt):
    # save path
    log_dir = opt.args.output_dir
    wdir = osp.join(log_dir, 'weights')  # weights directory
    os.makedirs(wdir, exist_ok=True)
    logger.info('ckpt is saved to {}'.format(wdir))

    last = osp.join(wdir, 'last.pt')
    best = osp.join(wdir, 'best.pt')
    results_file = osp.join(log_dir, 'train_results.txt')
    epochs, batch_size, rank = opt.TRAIN.END_EPOCH, opt.TRAIN.BATCH_SIZE, opt.global_rank

    # device and cudnn
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    # data information
    trainset_paths, valset_paths = parser_mot_train_data(opt)
    dataset_root = opt.TRAIN.DATASET.ROOT_DIR
    nc, names = (1, ['person'])     # TODO: only support 1 class here, please modify it to meet your needs
    logger.info('only support 1 class (people) for now')
    logger.info('training datasets: {}'.format(trainset_paths))
    logger.info('validation datasets: {}'.format(valset_paths))
    logger.info('dataset root directory: {}'.format(dataset_root))


    # save run settings
    # logger.info('save config file to {}'.format(log_dir))
    # with open(osp.join(log_dir, 'opt.yaml'), 'w') as f:
    #     opt.args = edict(vars(opt.args))
    #     yaml.dump(opt, f, sort_keys=False)

    # define model
    logger.info('build model: {}'.format(opt.MODEL.Name))
    model_builder = simple_mot_builder(opt)
    model = model_builder.build(pre_save_cfg=None).to(device)

    # load pretrained weights
    logger.info('load pretrained weights {}'.format(opt.TRAIN.PRETRAIN))
    if opt.TRAIN.PRETRAIN:
        if 'http' in opt.TRAIN.PRETRAIN or 'www' in opt.TRAIN.PRETRAIN:
            with torch_distributed_zero_first(rank):
                attempt_download(opt.TRAIN.PRETRAIN)  # download if not found locally

        if not osp.exists(opt.TRAIN.PRETRAIN):
            pretrain_path = osp.join('pretrain', opt.TRAIN.PRETRAIN)
            if not osp.exists(pretrain_path):
                pretrain_path = osp.join('../', pretrain_path)  # debug
        else:
            pretrain_path = opt.TRAIN.PRETRAIN

        ckpt = torch.load(pretrain_path, map_location=device)  # load checkpoint
        exclude = ['anchor'] if opt.args.cfg else []  # exclude keys

        if type(ckpt['model']).__name__ == "OrderedDict":
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['model'].float().state_dict()  # to FP32

        model_helper.check_keys(model, state_dict)
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.TRAIN.PRETRAIN))  # report
    else:
        logger.info('no pretrained model')

    # some infos
    net_max_stride = int(max(model.stride))   # model stride

    # train dataloader
    logger.info('build train data loader ...')
    dataloader, dataset = create_dataloader(dataset_root, trainset_paths, opt.TRAIN.IMG_SIZE, batch_size, net_max_stride, opt, augment=True,
                                            cache=opt.args.cache_images, rect=False, rank=rank,
                                            world_size=opt.world_size, workers=opt.args.workers, state="train")

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches

    # validation dataloader
    logger.info('build val data loader ...')
    if rank in [-1, 0]:
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        valloader = create_dataloader(dataset_root, valset_paths, opt.TRAIN.IMG_SIZE, batch_size, net_max_stride, opt, augment=False,
                                       cache=opt.args.cache_images, rect=True, rank=-1, world_size=opt.world_size, workers=opt.args.workers, state="val")[0]

    # build loss
    logger.info('build loss function ...')
    if opt.MODEL.Name in ['JDE', 'CSTrack', 'OMC']:
        mot_loss = JDELoss(dataset.nID, opt.MODEL.id_embedding)  # dataset.nID: number of ids
    else:
        raise ValueError('Unsupported Model Type for Building Loss.')

    # build optimizer
    accumulate = max(round(opt.TRAIN.UPDATE_BATCH / batch_size), 1)  # accumulate loss before optimizing
    opt.TRAIN.WEIGHT_DECAY *= batch_size * accumulate / opt.TRAIN.UPDATE_BATCH  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if opt.TRAIN.ADAM:
        base_optimizer = optim.Adam(pg0, lr=opt.TRAIN.LR_G0, betas=(opt.TRAIN.MOMENTUM, 0.999))  # adjust beta1 to momentum
        optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)
    else:
        optimizer = optim.SGD(pg0, lr=opt.TRAIN.LR_G0, momentum=opt.TRAIN.MOMENTUM, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': opt.TRAIN.WEIGHT_DECAY})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    optimizer.add_param_group({'params': mot_loss.parameters()})

    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if opt.TRAIN.PLOT_LR:
        logger.info('plot learning rate and save it to {}'.format(opt.args.output_dir))
        plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=opt.args.output_dir)

    # load pretrain
    start_epoch, best_fitness = 0, 0.0
    if opt.TRAIN.RESUME:   # resume
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.TRAIN.RESUME, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    start_epoch = opt.TRAIN.START_EPOCH

    # DP instead of DDP
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        logger.info('use DP instead of DDP training')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.TRAIN.SYNC_BN and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # exponential moving average trick
    ema = model_helper.ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.args.local_rank], output_device=(opt.args.local_rank), find_unused_parameters=True)

    # Model parameters
    opt.TRAIN.CLS_WEIGHT *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc                     # attach number of classes to model
    model.hyp = opt                   # attach hyperparameters to model
    model.gr = 1.0                    # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # class frequency
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        # plot_labels(labels, save_dir=log_dir)   # used for detection (multi-class)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

        if hasattr(opt, 'wandb_instance'):
            opt.wandb_instance.watch(model)


    # start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info('Image sizes (%g , %g)' % (opt.TRAIN.IMG_SIZE[0], opt.TRAIN.IMG_SIZE[1]))
    logger.info('Using %g dataloader workers' % dataloader.num_workers)
    logger.info('Starting training for %g epochs...' % epochs)

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):
        # freeze some layers
        logger.info('unfix layers: {}'.format(opt.TRAIN.FIX_LAYERS))
        if opt.TRAIN.FIX_LAYERS:
            if epoch < opt.TRAIN.UNFIX_EPOCH:
                for k, v in model.named_parameters():
                    if any(x in k for x in opt.TRAIN.FIX_LAYERS):
                        print('freezing %s' % k)
                        v.requires_grad = False

        model.train()

        # update image weights (optional)
        if dataset.image_weights:
            logger.info('update image weights')

            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

            # Broadcast if DDP
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)  # important for DDP

        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU','idloss', 'obj', 'total', 'targets', 'lr'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, opt.TRAIN.UPDATE_BATCH / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    if opt.TRAIN.FINETUNE:
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch) * 0.1])
                    else:
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, opt.TRAIN.MOMENTUM])

            for j, x in enumerate(optimizer.param_groups):
                if ni > nw and ni <= 20 * nb:
                    if opt.TRAIN.FINETUNE:
                        x['lr'] = x['initial_lr'] * 0.1
                    else:
                        x['lr'] = x['initial_lr']
                if ni > 20 * nb:
                    x['lr'] = x['initial_lr'] * 0.1
                lr_now = x['lr']

            # multi-scale training
            if opt.args.multi_scale and ni > nw and epoch % 2 == 0:
                sz = random.randrange(opt.TRAIN.IMG_SIZE[0] * 0.5, opt.TRAIN.IMG_SIZE[0] * 1.0) //net_max_stride * net_max_stride  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / net_max_stride) * net_max_stride for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Autocast
            with amp.autocast(enabled=cuda):
                # forward
                pred = model(imgs)   # pred[0]: [b, h//8, w//8, 512]
                                     # pred[1][0][0]:  [b, 3, h//8, w//8, 6]  box/cls/object-ness 0f stride 8
                                     # pred[1][0][1]:  [b, 3, h//16, w//16, 6]  box/cls/object-ness 0f stride 16
                                     # pred[1][0][2]:  [b, 3, h//32, w//32, 6]  box/cls/object-ness 0f stride 32
                # loss
                loss, loss_items = mot_loss(pred, targets.to(device), model)  # scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], lr_now)
                pbar.set_description(s)

                if hasattr(opt, 'wandb_instance'):
                    opt.wandb_instance.log({
                        "epoch": epoch,
                        "iter": i,
                        "lr": lr_now,
                        "Box-IoU Loss": mloss[0],
                        "Object-ness Loss": mloss[1],
                        "ID-embedding Loss": mloss[2],
                        "total-loss": mloss[3],
                        "targets-number": targets.shape[0],
                    })


                # plot
                # if ni < 3:
                #     f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                #     result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                #     if tb_writer and result is not None:
                #         tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == epochs
            if rank in [-1, 0]:  # Calculate mAP
                results, maps, times = eval.test(
                                                 batch_size=batch_size,
                                                 imgsz=opt.TRAIN.IMG_SIZE,
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 dataloader=valloader,
                                                 save_dir=log_dir,
                                                 opt=opt)

            # write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 4 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

            # tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            if hasattr(opt, 'wandb_instance'):
                opt.wandb_instance.log({
                    "Val-Precision": results[0],
                    "Val-REcall": results[1],
                    "map@0.5": results[2],
                    "map@0.5:0.95": results[3],
                })

            # update best mAP
            fi = jde_fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness: best_fitness = fi

            # save model
            logger.info('save model ...')
            if rank in [-1, 0]:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    args = parse_args()
    if args.cfg is not None:
        if not osp.isfile(args.cfg): args.cfg = osp.join('../', args.cfg)
        opt = edict(reader.load_yaml(args.cfg))
    else:
        raise Exception('Please set the config file for tracking test!')

    opt.args = args

    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # check and make experiment dir
    name = args.name if args.name else opt.MODEL.Name
    ex_father_dir = osp.join(opt.args.output_dir, name)
    if not osp.exists(ex_father_dir): os.makedirs(ex_father_dir)
    ex_name = flush_exname(ex_father_dir)
    opt.args.output_dir = osp.join(ex_father_dir, ex_name)
    if not osp.exists(opt.args.output_dir): os.makedirs(opt.args.output_dir)

    logger.add(osp.join(opt.args.output_dir, 'train.log'))
    logger.info('train log saved to {}'.format(osp.join(opt.args.output_dir, 'train.log')))

    logger.info('Training Settings: \n')
    logger.info(opt)

    # Resume
    logger.info('check resume ...')
    if opt.TRAIN.RESUME:
        last = model_helper.get_latest_run(opt.args.output_dir) if opt.TRAIN.RESUME == 'last' else opt.TRAIN.RESUME  # resume from most recent run
        if last:
            logger.info(f'Resuming training from {last}')
            opt.TRAIN.PRETRAIN = last
        else:
            logger.info('no resuming')

    # check git info
    logger.info('check git status')
    if opt.global_rank in [-1, 0]:
        git_helper.check_git_status()

    opt.TRAIN.IMG_SIZE.extend([opt.TRAIN.IMG_SIZE[-1]] * (2 - len(opt.TRAIN.IMG_SIZE)))  # extend to 2 sizes (train, test)

    # DDP setup
    logger.info('setup DDP envs')
    device = select_device(opt.args.device, batch_size=opt.TRAIN.BATCH_SIZE)
    opt.device = device

    # DDP mode
    if opt.args.local_rank != -1:
        assert torch.cuda.device_count() > opt.args.local_rank
        torch.cuda.set_device(opt.args.local_rank)
        device = torch.device('cuda', opt.args.local_rank)
        opt.device = device
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.TRAIN.BATCH_SIZE % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.TRAIN.BATCH_SIZE //= opt.world_size

    # Tensorboard setup
    tb_writer = None
    if opt.global_rank in [-1, 0]:
        logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % osp.join(opt.args.output_dir, 'tensorboard'))
        tb_path = Path(opt.args.output_dir, 'tensorboard')
        if not osp.exists(tb_path): os.makedirs(tb_path)

        logger.info('Tensorboard record is saved in: {}'.format(tb_path))
        tb_writer = SummaryWriter(log_dir=tb_path)  # runs/exp
        opt.tb_writer = tb_writer

    # wandb setup
    # try:
    #     logger.info('use wandb to watch training')
    #     my_hostname = socket.gethostname()
    #     my_ip = socket.gethostbyname(my_hostname)
    #     logger.info('Hostname: {}'.format(my_hostname))
    #     logger.info('IP: {}'.format(my_ip))
    #     notes = {my_ip: my_hostname}
    #
    #     wandb_instance = recorder.setup_wandb(opt, notes)
    #     wandb_context = wandb_instance if wandb_instance is not None else nullcontext()
    #     opt.wandb_instance = wandb_instance
    #     with wandb_context:
    #         epoch_train(opt)
    # except:
    #     epoch_train(opt)

    epoch_train(opt)

