# -*- coding:utf-8 _*-
"""
@author:fxw
@file: psenet.py
@time: 2020/06/10
"""
import torch
import torch.nn as  nn
import torch.nn.functional as F
import math
from .mobilenet import *
from .shufflenet import *
from .resnet import *
from .squeezenet import SqueezenetBase


class Psenet(nn.Module):
    def __init__(self, base_name, num_classes=7, pretrained=True, scale=1):
        super(Psenet, self).__init__()
        if ('mobile' in base_name):
            self.baseModel = mobilenet_v3_large(pretrained=pretrained)
            inchannel = 160
            outchannel = 80
            extrachannel1 = 24
            extrachannel2 = 40
            extrachannel3 = 80

        elif ('spa' in base_name):
            self.baseModel = mobilenet_v3_large(pretrained=pretrained)
            inchannel = 160
            outchannel = 80
            extrachannel1 = 24
            extrachannel2 = 40
            extrachannel3 = 80

        elif ('shuffle' in base_name):
            self.baseModel = shufflenet_v2_x1_0(pretrained=pretrained)
            inchannel = 464
            outchannel = 128
            extrachannel1 = 24
            extrachannel2 = 116
            extrachannel3 = 232

        elif ('squeeze' in base_name):
            self.baseModel = SqueezenetBase('squeezenet1_1', pretrained=pretrained)
            inchannel = 512
            outchannel = 128
            extrachannel1 = 64
            extrachannel2 = 128
            extrachannel3 = 256

        elif ('resnet' in base_name):
            self.baseModel = resnet50(pretrained=pretrained)
            inchannel = 2048
            outchannel = 256
            extrachannel1 = 256
            extrachannel2 = 512
            extrachannel3 = 1024

        # Top layer
        self.toplayer = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer_bn = nn.BatchNorm2d(outchannel)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(outchannel)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(outchannel)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(outchannel)
        self.smooth3_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(extrachannel3, outchannel, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(outchannel)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(extrachannel2, outchannel, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(outchannel)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(extrachannel1, outchannel, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(outchannel)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(4 * outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outchannel, num_classes, kernel_size=1, stride=1, padding=0)

        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='bilinear') + y
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        c2, c3, c4, c5 = self.baseModel(x)

        # Top-down
        p5 = self.toplayer(c5)
        p5 = self.toplayer_relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = self.latlayer1_relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.conv3(out)
        out = self._upsample(out, x, scale=self.scale)

        return out


