#-*- coding:utf-8 _*-
"""
@author:fxw
@file: SqueezeNet.py
@time: 2020/06/09
"""
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.group1 = nn.Sequential(
            OrderedDict([
                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                ('squeeze_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                ('expand1x1_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group3 = nn.Sequential(
            OrderedDict([
                ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
                ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x),self.group3(x)], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, padding=3,stride=2),
                nn.ReLU(inplace=True),
                Fire(96, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is final_conv:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        return x

def squeezenet1_0(pretrained=True, model_root=None, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['squeezenet1_0'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


def squeezenet1_1(pretrained=True, model_root=None, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['squeezenet1_1'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


class SqueezenetBase(nn.Module):
    def __init__(self,name,pretrained=True):
        super(SqueezenetBase, self).__init__()
        self.name = name
        if(name == 'squeezenet1_0'):
            self.basemodel = squeezenet1_0(pretrained=pretrained)
        elif(name == 'squeezenet1_1'):
            self.basemodel = squeezenet1_1(pretrained=pretrained)
        else:
            print('not support')
    def forward(self,x):
        if(self.name=='squeezenet1_0'):
            model_list = []
            i = 0
            for solution in self.basemodel.features.children():
                x = solution(x)
                if (i == 3 or i == 6 or i == 10 or i == 13):
                    model_list.append(x)
                i += 1
            p1 = model_list[0]
            p2 = model_list[1]
            p3 = model_list[2]
            p4 = model_list[3]
        elif(self.name=='squeezenet1_1'):
            model_list = []
            i = 0
            for solution in self.basemodel.features.children():
                x = solution(x)
                if (i == 2 or i == 5 or i == 8 or i == 13):
                    model_list.append(x)
                i += 1
            p1 = model_list[0]
            p2 = model_list[1]
            p3 = model_list[2]
            p4 = model_list[3]
        return p1,p2,p3,p4

# img = torch.ones(1,3,640,640)
# model = SqueezenetBase('squeezenet1_0')
# torch.save(model.state_dict(),'squeezenet.pth')
# p1,p2,p3,p4 = model(img)
# print(p1.shape)
# print(p2.shape)
# print(p3.shape)
# print(p4.shape)
