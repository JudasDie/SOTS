''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: siamese trainer
Data: 2021.6.23
'''

import math
import time
import torch
import pdb
import utils.log_helper as recorder
import utils.model_helper as loader
import torch.distributed as dist


def siamese_train(inputs):
    # parser inputs
    train_loader, model, optimizer, device = inputs['data_loader'], inputs['model'], inputs['optimizer'], inputs['device']
    epoch, cur_lr, cfg, writer_dict, logger, wandb_instance = inputs['epoch'], inputs['cur_lr'], inputs['config'], inputs['writer_dict'], \
                                                              inputs['logger'], inputs['wandb_instance']

    # recorder
    batch_time = recorder.AverageMeter()
    data_time = recorder.AverageMeter()
    losses = recorder.AverageMeter()
    cls_losses = recorder.AverageMeter()
    reg_losses = recorder.AverageMeter()
    end = time.time()
    template_example_images = []   # for wandb record input images
    search_example_images = []   # for wandb record input images

    # switch to train mode
    model.train()
    model = model.to(device)

    for iter, batchinfo in enumerate(train_loader):
        data_time.update(time.time() - end)

        # SiamFC/SiamDW
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
                        'nas_list_z': inputs['nas_list_z'], 'nas_list_x': inputs['nas_list_x'],
                        'nas_list_nlp': inputs['nas_list_nlp'],
                        'phrase_ids': batchinfo['phrase_ids'], 'phrase_attnmask': batchinfo['phrase_attnmask']}

        model_loss = model(model_inputs)
        cls_loss = torch.mean(model_loss['cls_loss'])
        # reg_loss = torch.mean(model_loss['reg_loss']) if 'reg_loss' in model_loss.keys() else None
        if cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            reg_loss = model_loss['reg_loss']
            reg_loss['loss_l1'] = torch.mean(reg_loss['loss_l1'])
            reg_loss['loss_iou'] = torch.mean(reg_loss['loss_iou'])
        else:
            reg_loss = torch.mean(model_loss['reg_loss']) if 'reg_loss' in model_loss.keys() else None

        if cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.REG_WEIGHT_L1 * reg_loss[
                'loss_l1'] + cfg.TRAIN.REG_WEIGHT_IOU * reg_loss['loss_iou']
        elif cfg.MODEL.NAME in ['CNNInMo', 'VLT_SCAR']:
            cen_loss = torch.mean(model_loss['cen_loss'])
            loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.REG_WEIGHT * reg_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        else:
            loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.REG_WEIGHT * reg_loss if reg_loss is not None else cls_loss
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        if cfg.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if loader.is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss = cls_loss.item()
        cls_losses.update(cls_loss, template.size(0))

        if cfg.MODEL.NAME in ['TransInMo', 'VLT_TT']:
            reg_loss = (reg_loss['loss_l1'] + reg_loss['loss_iou']).item()
        else:
            reg_loss = reg_loss.item() if reg_loss is not None else cls_loss
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            if not cfg.TRAIN.DDP.ISTRUE or dist.get_rank() == 0:
                logger.info(
                    'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_Loss:{cls_loss.avg:.5f} \t REG_Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                        epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                        loss=losses, cls_loss=cls_losses, reg_loss=reg_losses))

                recorder.print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                            cfg.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        writer.add_scalar('loss', loss, global_steps)
        writer.add_scalar('cls_loss', cls_loss, global_steps)
        writer.add_scalar('reg_loss', reg_loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        if wandb_instance is not None:
            # save some input images for watching
            # template_example_images.append(wandb_instance.Image(
            #     template[0], caption="Epoch:{}, Iteration:{}".format(epoch, iter)))
            #
            # search_example_images.append(wandb_instance.Image(
            #     search[0], caption="Epoch:{}, Iteration:{}".format(epoch, iter)))
            #
            wandb_instance.log({
                # "Template-Examples": template_example_images,   # save
                # "Search-Examples": search_example_images,   # save
                "epoch": epoch,
                "iter": iter,
                "lr": cur_lr,
                "Cla. Loss": cls_loss,
                "Reg. Loss": reg_loss,
                "loss": loss
            })


    return model, writer_dict

