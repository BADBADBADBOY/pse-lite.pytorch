"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: train_prune_finetune.py
@time: 2020/6/20 10:59

"""

import sys
sys.path.append('/home/aistudio/external-libraries')
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
import cv2
from torch.autograd import Variable
from torch.utils import data
import os
from dataset.dataloader import DataLoader
from utils.metrics import runningScore, cal_kernel_score, cal_text_score
import models
from utils.logger import Logger
from utils.misc import AverageMeter
from loss.loss import PseLoss
import time
from get_prune_model import load_prune_model
# from pse import pse

def train(train_loader, model, criterion, optimizer):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)

    end = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs = Variable(imgs.cuda())
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        training_masks = Variable(training_masks.cuda())

        outputs = model(imgs)
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1:, :, :]

        loss = criterion(texts, gt_texts, kernels, gt_kernels, training_masks)
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            output_log = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()

    return (
    losses.avg, score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU'])


def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


# def lr_poly(base_lr, iter, max_iter, power):
#     return base_lr*((1-float(iter)/max_iter)**(power))

# def adjust_learning_rate(base_lr, optimizer, i_iter):
#     args = self.args
#     lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
#     optimizer.param_groups[0]['lr'] = lr
#     return lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def set_seed(seed):
    import numpy as np
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLOBAL_WORKER_ID = None
GLOBAL_SEED = 1000


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def main(args):
    if args.checkpoint == '':
        args.checkpoint = "checkpoints/finetune_%s_bs_%d_ep_%d" % (args.backbone, args.batch_size, args.n_epoch)
    print('checkpoint path: %s' % args.checkpoint)
    print('init lr: %.8f' % args.lr)
    print('schedule: ', args.schedule)
    sys.stdout.flush()
    start_epoch  = 0
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    data_loader = DataLoader(is_transform=True, kernel_num=args.kernel_num, min_scale=args.min_scale)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=True)

    model = models.Psenet(args.backbone).cuda()
    model = load_prune_model(model,args.prune_model)

    criterion = PseLoss(kernel_num=args.kernel_num, text_loss_ratio=0.7).cuda()
    model = torch.nn.DataParallel(model)

    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    if args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=args.backbone, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=args.backbone)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.', 'Train IOU.', 'recall', 'precision', 'f1'])

    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_te_acc, train_ke_acc, train_te_iou, train_ke_iou = train(train_loader, model, criterion,optimizer)
        #         if (epoch < 200):
        recall, precision, f1 = 0, 0, 0
        #         else:
        #             recall, precision, f1 = test(model)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': args.lr,
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)
        logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc, train_te_iou, recall, precision, f1])
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', nargs='?', type=str, default='resnet')
    parser.add_argument('--img_size', nargs='?', type=int, default=640,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=120,
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80, 100],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--sr_lr', nargs='?', type=float, default=None,
                        help='sr Rate')
    parser.add_argument('--num_workers', nargs='?', type=int, default=0,
                        help='num workers to train')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='kernel_num to train')
    parser.add_argument('--min_scale', nargs='?', type=float, default=0.4,
                        help='min_scale to train')
    parser.add_argument('--resume', nargs='?', type=str,default='',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--prune_model', default='./pruned/checkpoints/pruned_dict.pth.tar', type=str, metavar='PATH',
                        help='pruned model path')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='checkpoint path')
    args = parser.parse_args()

    main(args)
