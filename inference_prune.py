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
import os
import torch
import cv2
import numpy as np
import time
from cal_rescall.script import cal_recall_precison_f1
from tqdm import tqdm
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
# from pse import pse
import models
import argparse
from get_prune_model import load_prune_model

binary_th = 1
kernel_num = 7
scale = 1
long_size = 2240
min_kernel_area = 5.0
min_area = 800.0
min_score = 0.93


def scale_img(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def debug(img_name, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    cv2.imwrite(output_root + img_name + '.jpg', res)


def write_result_as_txt(image_name, bboxes, path):
    filename = os.path.join(path, 'res_%s.txt' % (image_name))
    with open(filename, 'w+', encoding='utf-8') as fid:
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
            fid.write(line)


class Detect():
    def __init__(self, args):
        super(Detect, self).__init__()
        model = models.Psenet(args.backbone)
        model = load_prune_model(model,args.prune_checkpoint)
        pre_model = torch.load(args.checkpoint)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(pre_model['state_dict'])
        self.model = model
        self.model.eval()
        self.args = args

    def detect(self, img):
        text_box = img.copy()
        img = scale_img(img, long_size=long_size)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        img = Variable(img.cuda()).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img)
        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        cv2.imwrite('score.jpg', score * 255)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

        # c++ version pse
        pred = pse(kernels, min_kernel_area / (scale * scale))
        # python version pse
        # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))

        scale_im = (text_box.shape[1] * 1.0 / pred.shape[1], text_box.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < min_area / (scale * scale):
                continue
            score_i = np.mean(score[label == i])
            if score_i < min_score:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale_im
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        return bboxes, text_box

    def val(self, val_dir, gt_path, pre_path, scale=1):
        files = os.listdir(val_dir)

        self.model.eval()

        total_frame = 0.0
        total_time = 0.0
        bar = tqdm(total=len(files))
        for idx in range(len(files)):
            img = cv2.imread(os.path.join(val_dir, files[idx]))
            text_box = img.copy()
            img = scale_img(img, long_size=long_size)
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            sys.stdout.flush()
            bar.update(1)
            img = Variable(img.cuda()).unsqueeze(0)

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                outputs = self.model(img)
            score = torch.sigmoid(outputs[:, 0, :, :])
            outputs = (torch.sign(outputs - binary_th) + 1) / 2

            text = outputs[:, 0, :, :]
            kernels = outputs[:, 0:kernel_num, :, :] * text

            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

            # c++ version pse
            pred = pse(kernels, min_kernel_area / (scale * scale))
            # python version pse
            # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))

            scale_im = (text_box.shape[1] * 1.0 / pred.shape[1], text_box.shape[0] * 1.0 / pred.shape[0])
            label = pred
            label_num = np.max(label) + 1
            bboxes = []
            for i in range(1, label_num):
                points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
                if points.shape[0] < min_area / (scale * scale):
                    continue
                score_i = np.mean(score[label == i])
                if score_i < min_score:
                    continue

                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect) * scale_im
                bbox = bbox.astype('int32')
                bboxes.append(bbox.reshape(-1))

            torch.cuda.synchronize()
            end = time.time()
            total_frame += 1
            total_time += (end - start)
            #         print('fps: %.2f'%(total_frame / total_time))
            sys.stdout.flush()

            for bbox in bboxes:
                cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
            image_name = files[idx].split('.')[0]
            write_result_as_txt(image_name, bboxes, 'outputs/submit_ic15/')
            text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
            debug(image_name, [[text_box]], 'outputs/vis_ic15/')
        bar.close()
        sys.stdout.flush()
        result_dict = cal_recall_precison_f1(gt_path, pre_path)
        return result_dict


def test_img(args):
    detect_obj = Detect(args)
    # test
    img = cv2.imread(args.img_file)
    bboxes, img = detect_obj.detect(img)
    cv2.imwrite('./predict.jpg', img)


def val_img(args):
    detect_obj = Detect(args)
    result_dict = detect_obj.val('/src/notebooks/fxw20190611/PSENet-master/train_data/icdar2015/test/image',
                                 '/src/notebooks/fxw20190611/PSENet-master/train_data/icdar2015/test/label',
                                 './outputs/submit_ic15/')
    print(result_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', nargs='?', type=str, default='resnet')
    parser.add_argument('--max_size', nargs='?', type=int, default=2240,
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='kernel_num to train')
    parser.add_argument('--img_file',
                        default='/home/aistudio/work/data/icdar/test_img/img_10.jpg',
                        type=str,
                        help='')
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_resnet_bs_8_ep_120/checkpoint.pth.tar', type=str,
                        metavar='PATH',
                        help='path to save checkpoint')

    parser.add_argument('--prune_checkpoint', default='./pruned/checkpoints/pruned_dict.pth.tar',
                        type=str,
                        metavar='PATH',
                        help='prune checkpoint')
    args = parser.parse_args()

    test_img(args)
#     val_img(args)

