# -*- coding: future_fstrings -*-
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.residual_block import get_block
import torch.nn as nn

class ResSCNN(ME.MinkowskiNetwork):
  CHANNELS = [3, 64, 64, 64, 64]

  def __init__(self, bn_momentum=0.1, D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    CHANNELS = self.CHANNELS
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[0],
        out_channels=CHANNELS[1],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)

    self.block1 = get_block(
        CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)

    self.block2 = get_block(
        CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3], momentum=bn_momentum)

    self.block3 = get_block(
        CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4 = ME.MinkowskiBatchNorm(CHANNELS[4], momentum=bn_momentum)

    self.block4 = get_block(
        CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.glob_avg = nn.AdaptiveMaxPool2d((1, None))

    self.fc1 = nn.Linear(256, 32)
    self.fc2 = nn.Linear(32, 1)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = MEF.relu(out_s1)
    out1 = self.block1(out_s1)

    out1_ = self.glob_avg(out1.F.unsqueeze(0))

    out_s2 = self.conv2(out1)
    out_s2 = self.norm2(out_s2)
    out_s2 = MEF.relu(out_s2)
    out2 = self.block2(out_s2)

    out2_ = self.glob_avg(out2.F.unsqueeze(0))

    out_s3 = self.conv3(out2)
    out_s3 = self.norm3(out_s3)
    out_s3 = MEF.relu(out_s3)
    out3 = self.block3(out_s3)

    out3_ = self.glob_avg(out3.F.unsqueeze(0))

    out_s4 = self.conv4(out3)
    out_s4 = self.norm4(out_s4)
    out_s4 = MEF.relu(out_s4)
    out4 = self.block4(out_s4)

    out4_ = self.glob_avg(out4.F.unsqueeze(0))

    out = torch.cat((out1_, out2_, out3_, out4_), 2)

    out = self.fc1(out)
    out = self.fc2(out)

    return out




