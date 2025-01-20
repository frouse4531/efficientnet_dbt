# From "Learning Deep Bilinear Transformation for Fine-grained Image Represetation"
# https://dl.acm.org/doi/pdf/10.5555/3454287.3454672
#
# @incollection{NIPS2019_8680,
# title = {Learning Deep Bilinear Transformation for Fine-grained Image Representation},
# author = {Zheng, Heliang and Fu, Jianlong and Zha, Zheng-Jun and Luo, Jiebo},
# booktitle = {Advances in Neural Information Processing Systems 32},
# pages = {4279--4288},
# year = {2019}
#
# Original MXNet version https://github.com/researchmm/DBTNet
# Modifications of Pytorch implementation by https://github.com/wuwusky/DBT_Net
#

import torch
import numpy
import math
import os

import argparse
import math
import os
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
)

# from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms, datasets, models

import torch.nn as nn
from PIL import Image

# from softtriple import loss
# from softtriple.evaluation import evaluation
# from softtriple import net

import timm.data.auto_augment
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation


class loss_function:
    def __init__(self, criterion):
        self.criterion = criterion.cuda()

    def __call__(self, output, target):
        return self.criterion(output, target)


class dbt_loss_fcn(loss_function):
    def __init__(self, lam=1.0e-04):
        criterion = nn.CrossEntropyLoss().cuda()
        super().__init__(criterion)
        self.mse_loss = nn.MSELoss(reduction="sum").cuda()
        self.loss_sg = None
        self.lam = lam  # noted as lambda in paper

    def __call__(self, output, target):
        if self.loss_sg is None:
            return self.criterion(output, target)
        # TODO- just make 0 matrix of correct shape instead of self.loss_sg*0?
        loss_grouping = self.mse_loss(self.loss_sg, self.loss_sg * 0) * self.lam
        # print("loss grouping {}".format(loss_grouping))
        return self.criterion(output, target) + loss_grouping


class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, width, num_group):
        super(GroupConv, self).__init__()
        self.num_group = num_group
        self.in_channels = in_channels
        self.out_channels = out_channels
        # print("group conv init: in channels {} out channels {}".format(in_channels, out_channels))
        self.matrix_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        nn.init.constant_(self.matrix_conv.weight, 1.0)
        nn.init.constant_(self.matrix_conv.bias, 0.1)
        self.loss = 0

    def forward(self, x):
        channels = self.out_channels
        matrix_act = self.matrix_conv(x)
        matrix_act = self.bn(matrix_act)
        matrix_act = self.relu(matrix_act)

        tmp = matrix_act + 0.001
        b, c, w, h = tmp.shape
        width = w
        # print("b {} c {} w {} h {} in channels {} out channels {}".format(b, c, w, h, self.in_channels, channels))
        tmp = tmp.view(int((b * c * w * h) / (width * width)), width * width)
        tmp = F.normalize(tmp, p=2)
        tmp = tmp.view(b, channels, width * width)
        tmp = tmp.permute(1, 0, 2)
        tmp = tmp.reshape(channels, b * w * h)
        tmp_T = tmp.transpose(1, 0)
        co = tmp.mm(tmp_T)
        co = co.view(1, channels * channels)
        co = co / 128

        gt = torch.ones((self.num_group))
        gt = gt.diag()
        gt = gt.reshape((1, 1, self.num_group, self.num_group))
        gt = gt.repeat(
            (1, int((channels / self.num_group) * (channels / self.num_group)), 1, 1)
        )
        gt = F.pixel_shuffle(gt, upscale_factor=int(channels / self.num_group))
        gt = gt.reshape((1, channels * channels))
        if torch.cuda.is_available:
            device = torch.device("cuda")
            gt = gt.to(device)
        loss_single = torch.sum((co - gt) * (co - gt) * 0.001, dim=1)
        self.loss = loss_single / (
            (channels / 512.0) * (channels / 512.0)
        )  # loss_single.repeat(b)
        # print("matrix_act {} loss {}".format(matrix_act, loss))
        return matrix_act


class GroupBillinear(nn.Module):
    def __init__(self, num_group, width, channels):
        super(GroupBillinear, self).__init__()
        self.num_group = num_group
        self.num_per_group = int(channels / num_group)
        self.channels = channels
        self.fc = nn.Linear(channels, channels, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        # self.BL = nn.Bilinear(self.num_group, self.num_group, channels)

    def forward(self, x):
        b, c, w, h = x.shape
        width = w
        num_dim = b * c * w * h
        tmp = x.permute(0, 2, 3, 1)

        tmp = tmp.reshape(num_dim // self.channels, self.channels)
        my_tmp = self.fc(tmp)
        tmp = tmp + my_tmp

        tmp = tmp.reshape(
            ((num_dim // self.channels), self.num_group, self.num_per_group)
        )
        tmp_T = tmp.permute((0, 2, 1))

        # tmp = self.BL(tmp_T, tmp_T)
        # tmp = tmp.reshape((b, self.width, self.width, c))
        # tmp = tmp.permute((0,3,1,2))

        tmp = torch.tanh(torch.bmm(tmp_T, tmp) / 32)
        tmp = tmp.reshape((b, width, width, self.num_per_group * self.num_per_group))
        # tmp = F.upsample_bilinear(tmp, (width, c))
        tmp = F.interpolate(tmp, (width, c))
        tmp = tmp.permute((0, 3, 1, 2))

        out = x + self.bn(tmp)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplaces = inplanes
        self.planes = planes
        self.conv_ch = conv1x1(inplanes, planes, stride=1)

    def forward(self, x):
        if self.inplaces != self.planes:
            identity = self.conv_ch(x)
            identity = self.bn1(identity)
            identity = self.relu(identity)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        use_SG_GB=False,
        featuremap_size=0,
    ):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.downsample = downsample
        self.stride = stride
        self.use_SG_GB = use_SG_GB
        if self.use_SG_GB:
            self.SG = GroupConv(inplanes, planes, featuremap_size, 16)
            self.GB = GroupBillinear(16, featuremap_size, planes)
            self.conv1 = conv3x3(planes, planes)
        else:
            self.conv1 = conv1x1(inplanes, planes)

    def forward(self, x):
        identity = x

        if self.use_SG_GB:
            out = self.SG(x)
            out = self.GB(out)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_dim=4, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_sim = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out = nn.Sequential(
            nn.Linear(512 * block.expansion, output_dim),
            nn.Sigmoid(),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = F.dropout2d(x, p=0.25, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x


class ResNet_SG_GB(nn.Module):

    def __init__(
        self,
        block,
        nblock_copies,
        output_dim=4,
        as_encoder=False,
        zero_init_residual=True,
        down_1=False,
    ):
        super(ResNet_SG_GB, self).__init__()
        self.inplanes = 64
        self.featuremap_size = 224
        self.down_1 = down_1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        # self.conv1_sim = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        self.all_gconvs = []

        if down_1:
            self.layer1 = self._make_layer(block, 64, nblock_copies[0], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, nblock_copies[0])
            self.featuremap_size = int(self.featuremap_size * 0.5)
        self.layer2 = self._make_layer(block, 128, nblock_copies[1], stride=2)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        self.layer3 = self._make_layer_SG_GB(block, 256, nblock_copies[2], stride=2)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        self.layer4 = self._make_layer_SG_GB(block, 512, nblock_copies[3], stride=2)
        self.featuremap_size = int(self.featuremap_size * 0.5)

        self.SG_end = GroupConv(
            512 * block.expansion, 512 * block.expansion, self.featuremap_size, 32
        )
        self.all_gconvs.append(self.SG_end)
        self.GB_end = GroupBillinear(32, self.featuremap_size, 512 * block.expansion)
        self.bn_end = nn.BatchNorm2d(512 * block.expansion)
        self.mse_loss = nn.MSELoss(reduction="none").cuda()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if as_encoder:
            self.out = nn.Sequential(
                nn.Linear(512 * block.expansion, output_dim),
                nn.ReLU(),
            )
        else:
            self.out = nn.Linear(512 * block.expansion, output_dim)
            # self.out = nn.Sequential(
            #     nn.Linear(512 * block.expansion, output_dim),
            #     nn.Sigmoid(),
            # )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_SG_GB(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        my_block = block(
            self.inplanes, planes, stride, downsample, True, self.featuremap_size
        )
        layers.append(my_block)
        self.all_gconvs.append(my_block.SG)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            my_block = block(self.inplanes, planes, 1, None, True, self.featuremap_size)
            layers.append(my_block)
            self.all_gconvs.append(my_block.SG)
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("res input size {}".format(x.shape))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)

        x = self.SG_end(x)
        x = self.GB_end(x)
        x = self.bn_end(x)

        x = self.avgpool(x)
        # x = F.dropout2d(x, p=0.25, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        cnt = 0
        for sg in self.all_gconvs:
            if cnt == 0:
                loss = sg.loss
            else:
                loss = loss + sg.loss
            cnt = cnt + 1
        loss_sg = loss / cnt

        return x, self.mse_loss(loss_sg, 0 * loss_sg)


def create_model(output_dim, as_encoder):
    """
    Create DBT model.  The output dimension is the number of units
    in the final fully connected layer.  If training as a classifier, this
    will be the number of classes.  If training as an encoder, it will be
    whatever encoding dimension we need.

    If we are training as a classifier, the last layer will be a sigmoid
    activation function.  If training as an encoder, we will use RELU to start
    with - a future modification might make this specified by the user.
    """
    model = ResNet_SG_GB(Bottleneck, (3, 4, 6, 3), output_dim, as_encoder)
    return model


class CosineAnnealingWithStepDecay:
    def __init__(self, gamma, decay_epoch, lr_min=None, t_mult=1):
        self.gamma = gamma
        self.decay_epoch = max(decay_epoch, 1)
        self.ti = self.decay_epoch
        self.tnext = None
        self.eta_min = 0 if lr_min is None else lr_min
        self.t_mult = t_mult

    # Return the multiplicative factor for the initial learning rate
    def step(self, epoch):
        n_steps = epoch // self.decay_epoch

        # Compute current time step.  Initially, just on [0, decay_epoch).  Then it becomes
        # time since last reset.  When specify mult factor for periodicity, need to
        # distinguish between decay epoch and current periodicity to compute time
        # since last reset.
        self.tnext = (
            epoch % self.ti if self.tnext is None else (self.tnext + 1) % self.ti
        )

        # Increase the peridicity on reset if multiplicative factor is 2 or more
        # Note on epoch 0, we do not increase the period because n_steps will be 0
        if self.tnext == 0 and n_steps > 0 and self.t_mult > 1:
            self.ti *= self.t_mult

        # first determine where on the eta_min to 1 oart if the cosine curve
        cos_factor = self.eta_min + 0.5 * (1 - self.eta_min) * (
            1.0 + math.cos(self.tnext * math.pi / self.ti)
        )
        # Now include how many step decays we have had
        next_lr_factor = cos_factor * self.gamma**n_steps
        # print(
        #     "T_cur {} Ti {} next factor {} cos factor {}".format(
        #         self.tnext, self.ti, next_lr_factor, cos_factor
        #     )
        # )
        return next_lr_factor


def lr_scheduler(
    optimizer, lr, lr_gamma, decay_epoch, max_epoch, name="exp", lr_min=None, t_mult=1
):
    """
    Create a learning rate scheduler.
    optimizer: pytorch optimizer
    lr: initial learning rate
    lr_gamma: learning rate decay factor.
    decay_epoch: for periodic style schedulers, the number of epochs before lr should decay by factor of gamma.
    max_epochs: total number of epochs for training run.
    lr_min: if specified, scheduler will be arranged to have lr decay to this value at max_epochs steps.
    name: name of scheduler to use.  Allowed names
        exp - ExponentialLR
        step - StepLR
        cos - cosine annealing scheduler with warm restarts.
        cosd - cosine annealing scheduler with warm restarts and step decay.
        none - no learning rate scheduler.
    """
    gamma = lr_gamma
    decay_epoch = max(int(decay_epoch), 1)  # make sure decay_epoch is an integer > 0
    if name == "exp":
        if lr_min is not None:
            decay_epoch = max_epoch - 1
            gamma = lr_min / lr

        # Set f such that after decay_epoch, we will have reduced the lr by
        # factor gamma
        f = math.exp(math.log(gamma) / decay_epoch)
        scheduler = ExponentialLR(optimizer, gamma=f)
    elif name == "step":
        if lr_min is not None:
            decay_epoch = max(decay_epoch, 2)
            n_step = max((max_epoch - 1) // decay_epoch, 1)
            gamma = math.exp(math.log(lr_min / lr) / n_step)
            decay_epoch = max_epoch - 1
        scheduler = StepLR(optimizer, decay_epoch, gamma=gamma)
    elif name == "cos":
        eta_min = 0 if lr_min is None else lr_min
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, decay_epoch, T_mult=t_mult, eta_min=eta_min
        )
    elif name == "cosd":
        scheduler = LambdaLR(
            optimizer,
            CosineAnnealingWithStepDecay(
                gamma, decay_epoch, lr_min=lr_min, t_mult=t_mult
            ).step,
        )
    elif name == "none":
        scheduler = None
    else:
        raise Exception("Scheduler name {} is not allowed!".format(name))
    return scheduler


# In original, keeping only until clear we have working code
#
# temp_model = ResNet_SG_GB(Bottleneck, (3,4,6,3), 5)

# print(temp_model.parameters)

# temp_input = torch.randn(5, 3, 224, 224)
# temp_label = torch.empty(5, dtype=torch.long).random_(3)

# loss_fun = nn.CrossEntropyLoss()
# loss_fun_2 = nn.MSELoss(reduction='mean')

# temp_model.zero_grad()
# temp_out, temp_loss = temp_model(temp_input)


# loss = loss_fun(temp_out, temp_label)
# temp_loss_ = temp_loss*0
# loss_matrix = loss_fun_2(temp_loss, temp_loss_)*1e-4

# loss_all = loss + loss_matrix
# print("loss all {}".format(loss_all))
# loss_all.backward()


# temp_model_SG = GroupConv(in_channels=64, out_channels=64, width=128, num_group=8)
# temp_model_GB = GroupBillinear(num_group=8, width=128, channels=64)
# temp_input = torch.randn(2, 64, 128, 128)
# temp_out = temp_model_SG(temp_input)
# out = temp_model_GB(temp_out)

# print('ok')
