#-*- coding:utf-8 _*-
"""
@author:fxw
@file: dataloader.py
@time: 2020/06/17
"""
#-*- coding:utf-8 _*-
"""
@author:fxw
@file: dataload.py
@time: 2020/04/28
"""
import numpy as np
from PIL import Image
from torch.utils import data
import glob
import cv2
import random
import os
import torchvision.transforms as transforms
import torch
from .random_thansform import Random_Augment
from .gen_map import gen_train_map

root_dir = '/home/aistudio/work/data/icdar/'
train_data_dir = root_dir + 'train_img'
train_gt_dir =  root_dir + 'train_gt'

radom_angle = (-10, 10)
random.seed(123456)

def get_img(img_path):
    img = cv2.imread(img_path)
    return img

def get_bboxes(gt_path):
    with open(gt_path,'r',encoding='utf-8') as fid:
        lines = fid.readlines()
    polys = []
    tags = []
    for line in lines:
        line = line.replace('\ufeff','').replace( '\xef\xbb\xbf','')
        gt = line.split(',')
        if "#" in gt[-1]:
            tags.append(True)
        else:
            tags.append(False)
        box = [int(gt[i]) for i in range(8)]
        polys.append(box)
    return np.array(polys), tags

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

class DataLoader(data.Dataset):
    def __init__(self, is_transform=True, kernel_num=7, min_scale=0.4):
        self.is_transform = is_transform
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        self.ra = Random_Augment()
        img_paths = glob.glob(os.path.join(train_data_dir,'*.jpg'))
        gt_paths = []
        for img_path in img_paths:
            im_name = img_path.split('/')[-1].split('.')[0]
            gt_paths.append(os.path.join(train_gt_dir,'gt_'+im_name+'.txt'))
        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        polys, dontcare = get_bboxes(gt_path)

        # if self.is_transform:
        #     img, polys = self.ra.random_scale(img, polys, img.shape[0])
        #     img, polys = self.ra.random_rotate(img, polys,radom_angle)
        #     img, polys = self.ra.random_flip(img, polys)
        #     img, polys, dontcare = self.ra.random_crop_db(img, polys, dontcare)
        # img,training_mask,gt_text, gt_kernels = gen_train_map(img, polys, dontcare,self.kernel_num,self.min_scale)
        

        if self.is_transform:
            img, polys = self.ra.random_scale(img, polys, img.shape[0])
            img, polys = self.ra.random_rotate(img, polys,radom_angle)
            img, polys = self.ra.random_flip(img, polys)   
        img,training_mask,gt_text, gt_kernels = gen_train_map(img, polys, dontcare,self.kernel_num,self.min_scale)
        imgs = [img, gt_text, training_mask]
        imgs.extend(gt_kernels)
        imgs = self.ra.random_crop_pse(imgs)
        img, gt_text, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        cv2.imwrite('img.jpg',img)
        cv2.imwrite('label.jpg',gt_text*255)
        cv2.imwrite('mask.jpg',training_mask*255)

        gt_kernels = np.array(gt_kernels)

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()
        # '''

        return img, gt_text, gt_kernels, training_mask


