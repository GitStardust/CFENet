# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:41:58 2019
@author: Administrator
MobileNet-V1
"""
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from thop import profile

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        # 标准卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 深度卷积
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 网络模型声明
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
            # nn.Conv2d(1024, 512, 3, 1, 1, groups=512, bias=False),
            # nn.Conv2d(4, 8, 3, 1, 1, groups=1, bias=False),
            # nn.Conv2d(4, 100, 3, 1, 1, groups=2, bias=False),
        )

        self.fc = nn.Linear(1024, 1000)

    # 网络的前向过程
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


# 速度评估
def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224).cpu()
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()

    print('%10s : %f' % (name, t3 - t2))


def make_moiblenet():
    # 标准卷积
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))

    # 深度卷积
    def conv_dw(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))

    # 网络模型声明
    layers=[
        conv_bn(3, 32, 2),
        conv_dw(32, 64, 1),
        conv_dw(64, 128, 2),
        conv_dw(128, 128, 1),
        conv_dw(128, 256, 2),
        conv_dw(256, 256, 1),
        conv_dw(256, 512, 2),
        conv_dw(512, 512, 1),
        conv_dw(512, 512, 1),
        conv_dw(512, 512, 1),
        conv_dw(512, 512, 1),
        conv_dw(512, 512, 1),
        conv_dw(512, 1024, 2),
        conv_dw(1024, 1024, 1),
        # nn.AvgPool2d(7),

    ]
    return layers
    # self.fc = nn.Linear(1024, 1000)



if __name__ == '__main__':
    # resnet18 = models.resnet18().cpu()
    # print(resnet18)
    # alexnet = models.alexnet().cpu()
    # vgg16 = models.vgg16().cpu()
    # squeezenet = models.squeezenet1_0().cpu()
    mobilenet = MobileNet().cpu()
    print(mobilenet)

    print('# mobilenet parameters:', sum(param.numel() for param in mobilenet.parameters()))
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(mobilenet, inputs=(input,))
    print("flops：",flops)
    print("params：",params)

    # speed(resnet18, 'resnet18')
    # speed(alexnet, 'alexnet')
    # speed(vgg16, 'vgg16')
    # speed(squeezenet, 'squeezenet')
    # speed(mobilenet, 'mobilenet')
