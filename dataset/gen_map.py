#-*- coding:utf-8 _*-
"""
@author:fxw
@file: gen_map.py
@time: 2020/06/17
"""

import numpy as np
from PIL import Image
import cv2
import random
import pyclipper
import Polygon as plg

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)

def gen_train_map(img, bboxes, tags,kernel_num=7,min_scale=0.4):
    bboxes =np.array(bboxes).astype(np.int)
    gt_text = np.zeros(img.shape[0:2], dtype='uint8').copy()
    training_mask = np.ones(img.shape[0:2], dtype='uint8').copy()

    if bboxes.shape[0] > 0:
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_text, [bboxes[i]], -1, 1, -1)
            if tags[i]:
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
    gt_kernels = []
    for i in range(1, kernel_num):
        rate = 1.0 - (1.0 - min_scale) / (kernel_num - 1) * i
        gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes, rate)
        for j in range(bboxes.shape[0]):
            cv2.drawContours(gt_kernel, [kernel_bboxes[j]], -1, 1, -1)
        gt_kernels.append(gt_kernel)
    return img,training_mask,gt_text,gt_kernels