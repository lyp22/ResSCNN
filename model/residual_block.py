import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class BasicBlock(nn.Module):
  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3):
    super(BasicBlock, self).__init__()

    self.conv1 = ME.MinkowskiConvolution(
        inplanes, planes, kernel_size=3, stride=stride, dimension=D)
    self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

    self.conv2 = ME.MinkowskiConvolution(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        bias=False,
        dimension=D)
    self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = MEF.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = MEF.relu(out)

    return out

def get_block(inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
    return BasicBlock(inplanes, planes, stride, dilation, downsample, bn_momentum, D)