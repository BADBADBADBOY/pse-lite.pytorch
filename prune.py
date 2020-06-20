"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: tt.py
@time: 2020/6/20 10:51

"""

import models
import torch
import torch.nn as  nn
import numpy as np
import collections
import torchvision.transforms as transform
import cv2
import os
import argparse

def prune(args):
    model = models.Psenet(args.backbone).cuda()
    checkpoint = torch.load(args.checkpoint)
    d = collections.OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        d[tmp] = value
    model.load_state_dict(d)


    bn_weights = []
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weights.append(m.weight.data.abs().clone())
    bn_weights = torch.cat(bn_weights, 0)

    sort_result, sort_index = torch.sort(bn_weights)

    thresh_index = int(args.cut_percent * bn_weights.shape[0])

    if (thresh_index == bn_weights.shape[0]):
        thresh_index = bn_weights.shape[0] - 1

    prued = 0
    prued_mask = []
    bn_index = []
    remain_channel_nums = []
    for k, m in enumerate(model.modules()):
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weight = m.weight.data.clone()
            mask = bn_weight.abs().gt(sort_result[thresh_index])
            remain_channel = mask.sum()

            if (remain_channel == 0):
                remain_channel = 1
                mask[int(torch.argmax(bn_weight))] = 1

            v = 0
            n = 1
            if (remain_channel % args.base_num != 0):
                if (remain_channel > args.base_num):
                    while (v < remain_channel):
                        n += 1
                        v = args.base_num * n
                    if (remain_channel - (v - args.base_num) < v - remain_channel):
                        remain_channel = v - args.base_num
                    else:
                        remain_channel = v
                    if (remain_channel > bn_weight.size()[0]):
                        remain_channel = bn_weight.size()[0]
                    remain_channel = torch.tensor(remain_channel)
                    result, index = torch.sort(bn_weight)
                    mask = bn_weight.abs().ge(result[-remain_channel])

            remain_channel_nums.append(int(mask.sum()))
            prued_mask.append(mask)
            bn_index.append(k)
            prued += mask.shape[0] - mask.sum()

    print(remain_channel_nums)
    print('total_prune_ratio:', float(prued) / bn_weights.shape[0])

    new_model = models.Psenet(args.backbone).cuda()

    index_bn = 0
    index_conv = 0
    cat_list_bn = [2, 12, 25, 44]
    cat_list_conv = [4, 14, 27, 46]  # downsample 位置
    cat_extre = list(range(53, 61))
    step_po = [52, 42, 23, 10]  # 每个尺度的位置 [10,23,42,52]
    layer_po = [53, 57, 58, 59]  # 每个smooth位置
    smooth_po = [54, 55, 56]  # 每个smooth位置
    out_po = [60]

    bn_mask = []
    conv_in_mask = []
    conv_out_mask = []

    for m in new_model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            pass
            if (index_bn in cat_list_bn):
                new_mask = prued_mask[index_bn + 1] | prued_mask[index_bn + 2]
                prued_mask[index_bn + 1] = new_mask
                prued_mask[index_bn + 2] = new_mask
            bn_mask.append(prued_mask[index_bn])
            m.num_features = prued_mask[index_bn].sum()
            index_bn += 1
        elif (isinstance(m, nn.Conv2d)):
            pass
            if (index_conv in cat_list_bn):
                new_mask = prued_mask[index_conv + 1] | prued_mask[index_conv + 2]
                prued_mask[index_conv + 1] = new_mask
                prued_mask[index_conv + 2] = new_mask
            if (index_conv in cat_list_conv):
                m.in_channels = prued_mask[index_conv - 4].sum()
                conv_in_mask.append(prued_mask[index_conv - 4])
            elif (index_conv == 0):
                m.in_channels = 3
                conv_in_mask.append(torch.ones(3))
            elif (index_conv in layer_po):
                index = layer_po.index(index_conv)
                m.in_channels = prued_mask[step_po[index]].sum()
                conv_in_mask.append(prued_mask[step_po[index]])
            elif (index_conv in smooth_po):
                index = smooth_po.index(index_conv)
                m.in_channels = prued_mask[layer_po[index + 1]].sum()
                conv_in_mask.append(prued_mask[layer_po[index + 1]])
            else:
                m.in_channels = prued_mask[index_conv - 1].sum()
                conv_in_mask.append(prued_mask[index_conv - 1])
            m.out_channels = prued_mask[index_conv].sum()
            conv_out_mask.append(prued_mask[index_conv])
            index_conv += 1

        if (index_bn > len(bn_index) - 1):
            break
    ###############################################

    index_bn = 0
    index_conv = 0
    cat_conv = [3, 4, 7, 10]
    cat_conv1 = [5, 8, ]

    cat_conv_1 = [13, 14, 17, 20, 23]
    cat_conv1_1 = [11, 15, 18, 21]

    cat_conv_2 = [26, 27, 30, 33, 36, 39, 42]
    cat_conv1_2 = [24, 28, 31, 34, 37, 40]

    cat_conv_3 = [45, 46, 49, 52]
    cat_conv1_3 = [43, 47, 50]

    for m in new_model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            if (index_bn in cat_conv):
                new_mask_mask = prued_mask[3] | prued_mask[4] | prued_mask[7] | prued_mask[10]
                m.num_features = new_mask_mask.sum()
                bn_mask[index_bn] = new_mask_mask
            elif (index_bn in cat_conv_1):
                new_mask_mask = prued_mask[13] | prued_mask[14] | prued_mask[17] | prued_mask[20] | prued_mask[23]
                m.num_features = new_mask_mask.sum()
                bn_mask[index_bn] = new_mask_mask
            elif (index_bn in cat_conv_2):
                new_mask_mask = prued_mask[26] | prued_mask[27] | prued_mask[30] | prued_mask[33] | \
                                prued_mask[36] | prued_mask[39] | prued_mask[42]

                m.num_features = new_mask_mask.sum()
                bn_mask[index_bn] = new_mask_mask
            elif (index_bn in cat_conv_3):
                new_mask_mask = prued_mask[45] | prued_mask[46] | prued_mask[49] | prued_mask[52]
                m.num_features = new_mask_mask.sum()
                bn_mask[index_bn] = new_mask_mask
            index_bn += 1
        elif (isinstance(m, nn.Conv2d)):
            if (index_conv in cat_conv):
                new_mask_mask = prued_mask[3] | prued_mask[4] | prued_mask[7] | prued_mask[10]
                m.out_channels = new_mask_mask.sum()
                conv_out_mask[index_conv] = new_mask_mask
            elif (index_conv in cat_conv_1):
                new_mask_mask = prued_mask[13] | prued_mask[14] | prued_mask[17] | prued_mask[20] | prued_mask[23]
                m.out_channels = new_mask_mask.sum()
                conv_out_mask[index_conv] = new_mask_mask
            elif (index_conv in cat_conv_2):
                new_mask_mask = prued_mask[26] | prued_mask[27] | prued_mask[30] | prued_mask[33] | \
                                prued_mask[36] | prued_mask[39] | prued_mask[42]

                m.out_channels = new_mask_mask.sum()
                conv_out_mask[index_conv] = new_mask_mask
            elif (index_conv in cat_conv_3):
                new_mask_mask = prued_mask[45] | prued_mask[46] | prued_mask[49] | prued_mask[52]
                m.out_channels = new_mask_mask.sum()
                conv_out_mask[index_conv] = new_mask_mask

            elif index_conv in cat_conv1 and index_conv > 1:
                m.in_channels = bn_mask[index_conv - 1].sum()
                conv_in_mask[index_conv] = bn_mask[index_conv - 1]

            elif index_conv in cat_conv1_1 and index_conv > 1:
                m.in_channels = bn_mask[index_conv - 1].sum()
                conv_in_mask[index_conv] = bn_mask[index_conv - 1]

            elif index_conv in cat_conv1_2 and index_conv > 1:
                m.in_channels = bn_mask[index_conv - 1].sum()
                conv_in_mask[index_conv] = bn_mask[index_conv - 1]

            elif index_conv in cat_conv1_3 and index_conv > 1:
                m.in_channels = bn_mask[index_conv - 1].sum()
                conv_in_mask[index_conv] = bn_mask[index_conv - 1]
            index_conv += 1

    index_conv = 0
    for m in new_model.modules():
        if (isinstance(m, nn.Conv2d)):
            if (index_conv in cat_list_conv):
                new_mask_mask = bn_mask[index_conv - 4]
                m.in_channels = new_mask_mask.sum()
                conv_in_mask[index_conv] = new_mask_mask
            index_conv += 1

    ###############################################
    index_bn = 0
    index_conv = 0
    for m in new_model.modules():
        if (isinstance(m, nn.Conv2d)):
            if (index_conv in layer_po):
                index = layer_po.index(index_conv)
                conv_in_mask[index_conv] = bn_mask[step_po[index]]
                m.in_channels = conv_in_mask[index_conv].sum()
            index_conv += 1

        if (index_bn > len(bn_index) - 1):
            break

    ##############################################
    index_bn = 0
    index_conv = 0
    cat_conv = [53, 54, 55, 56, 57, 58, 59]
    for m in new_model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            if (index_bn in cat_conv):
                new_mask_mask = bn_mask[53] | bn_mask[54] | bn_mask[55] | bn_mask[56] | bn_mask[57] | bn_mask[58] | \
                                bn_mask[
                                    59]
                m.num_features = new_mask_mask.sum()
                bn_mask[index_bn] = new_mask_mask
            index_bn += 1
        elif (isinstance(m, nn.Conv2d)):
            if (index_conv in cat_conv):
                new_mask_mask = bn_mask[53] | bn_mask[54] | bn_mask[55] | bn_mask[56] | bn_mask[57] | bn_mask[58] | \
                                bn_mask[
                                    59]
                m.out_channels = new_mask_mask.sum()
                conv_out_mask[index_conv] = new_mask_mask
            if (index_conv in smooth_po):
                new_mask_mask = bn_mask[53] | bn_mask[54] | bn_mask[55] | bn_mask[56] | bn_mask[57] | bn_mask[58] | \
                                bn_mask[
                                    59]
                m.in_channels = new_mask_mask.sum()
                conv_in_mask[index_conv] = new_mask_mask
            elif (index_conv in out_po):
                new_mask_mask = bn_mask[53] | bn_mask[54] | bn_mask[55] | bn_mask[56] | bn_mask[57] | bn_mask[58] | \
                                bn_mask[
                                    59]
                m.in_channels = 4 * new_mask_mask.sum()
                conv_in_mask[index_conv] = [new_mask_mask, new_mask_mask, new_mask_mask, new_mask_mask]
                pass
            index_conv += 1

    chann = [int(item.sum()) for item in bn_mask]
    print('new_prune_ratio', float(np.sum(chann)) / bn_weights.shape[0])
    print(chann)

    bn_i = 0
    conv_i = 0
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if (bn_i > 60):
            if isinstance(m0, nn.Conv2d):
                m1.in_channels = conv_out_mask[conv_i - 1].sum()
                idx0 = np.squeeze(np.argwhere(np.asarray(conv_out_mask[conv_i - 1].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(torch.ones(7).cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w = m0.weight.data[:, idx0, :, :].clone()
                m1.weight.data = w[idx1, :, :, :].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
        else:
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(bn_mask[bn_i].cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                bn_i += 1
            elif isinstance(m0, nn.Conv2d):
                if (isinstance(conv_in_mask[conv_i], list)):
                    idx0 = np.squeeze(np.argwhere(np.asarray(torch.cat(conv_in_mask[conv_i], 0).cpu().numpy())))
                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_mask[conv_i].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_mask[conv_i].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w = m0.weight.data[:, idx0, :, :].clone()
                m1.weight.data = w[idx1, :, :, :].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                conv_i += 1

    print(new_model)
    save_obj = {'prued_mask': prued_mask, 'bn_index': bn_index, 'state_dict': new_model.state_dict()}
    torch.save(save_obj, os.path.join(args.save_prune_model_path,'pruned_dict.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', nargs='?', type=str, default='resnet')

    parser.add_argument('--num_workers', nargs='?', type=int, default=0,
                        help='num workers to train')
    parser.add_argument('--base_num', nargs='?', type=int, default=8,
                        help='Base after Model Channel Clipping')
    parser.add_argument('--cut_percent', nargs='?', type=float, default=0.5,
                        help='Model channel clipping scale')
    parser.add_argument('--checkpoint', default='./checkpoint.pth.tar', type=str, metavar='PATH',
                        help='ori model path')
    parser.add_argument('--save_prune_model_path', default='./pruned/checkpoints/', type=str, metavar='PATH',
                        help='pruned model path')
    args = parser.parse_args()

    prune(args)
