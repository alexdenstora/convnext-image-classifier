import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import random


class LayerNorm2d(nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      #TODO
      self.layer_norm = nn.LayerNorm(in_channels)


    def forward(self, x: Tensor) -> Tensor:
      #TODO
      x = x.permute(0, 2, 3, 1)

      x = self.layer_norm(x)

      x = x.permute(0, 3, 1, 2)
      return x

class ConvNextStem(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()
    # TODO
    # conv2d args: in_channels, out_channels, kernel_size, stride
    self.patchify = nn.Conv2d(in_channels, out_channels, kernel_size, stride=kernel_size)
    self.layer_norm = LayerNorm2d(out_channels)
  
   def forward(self,x):
    # TODO
    x = self.patchify(x)
    x = self.layer_norm(x)
    return x

class ConvNextBlock(nn.Module):

  def __init__(self, d_in, layer_scale=1e-6, kernel_size=7, stochastic_depth_prob=1):
    super().__init__()
    #TODO
    padding = (kernel_size-1)//2
    self.depthwise = nn.Conv2d(in_channels=d_in, out_channels=d_in, kernel_size=kernel_size, padding=padding, groups=d_in)

    self.pointwise_expansion = nn.Conv2d(in_channels=d_in, out_channels=4*d_in, kernel_size=1)

    self.pointwise_contraction = nn.Conv2d(in_channels=4*d_in, out_channels=d_in, kernel_size=1)

    self.layer_norm = LayerNorm2d(d_in)

    self.activation = nn.GELU()

    self.scale = nn.Parameter(layer_scale*torch.ones((d_in, 1, 1)))

    self.stoch_prob = stochastic_depth_prob

  def forward(self,x):
    #TODO
    residual = x
    # spatial mixing
    x = self.depthwise(x)
    # normalization
    x = self.layer_norm(x)
    
    # bottle neck
    x = self.pointwise_expansion(x)
    x = self.activation(x)
    x = self.pointwise_contraction(x)

    # final output
    if self.training:
      surv = random.uniform(0,1)
      if surv <= self.stoch_prob:
        # x = x.permute(0, 2, 3, 1)
        x = residual + self.scale * x
        # x = x.permute(0, 3, 1, 2)
      else:
        x = residual
    else:
      # x =x.permute(0,2,3,1)
      x = residual + self.scale * x
      # x = x.permute(0, 3, 1, 2)
    return x
      


class ConvNextDownsample(nn.Module):
  def __init__(self, d_in, d_out, width=2):
    super().__init__()
    #TODO
    self.layer_norm = LayerNorm2d(d_in)
    self.conv = nn.Conv2d(in_channels=d_in, out_channels=d_out, kernel_size=width, stride=width)

  def forward(self,x):
    #TODO
    x = self.layer_norm(x)
    x = self.conv(x)
    return x

class ConvNextClassifier(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    #TODO
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.stand_norm = nn.LayerNorm(d_in)
    self.linear = nn.Linear(d_in, d_out)

  def forward(self,x):
    #TODO
    x = self.global_pool(x)
    x = self.stand_norm(x)
    x = nn.Flatten(x, 1)
    x = self.linear(x)
    return x


class ConvNext(nn.Module):

  def __init__(self, in_channels, out_channels, blocks=[96]):
    super().__init__()
    #TODO
    L = len(blocks)
    block_index = 0
    self.layers = nn.ModuleList()
    # create a new list of stages
    stages = [blocks[0]]
    for b in blocks:
      if b != stages[-1]:
        stages.append(b)
    # stages = 64, 128, 256, 512
    self.layers.append(ConvNextStem(in_channels, stages[0])) # stem
    self.layers.append(ConvNextBlock(stages[0], stochastic_depth_prob=(1 - block_index / L * 0.5))) # residual
    block_index += 1
    self.layers.append(ConvNextBlock(stages[0], stochastic_depth_prob=(1 - block_index / L * 0.5))) # residual
    block_index += 1
    
    for i in range(1, len(stages)):
      self.layers.append(ConvNextDownsample(stages[i-1], stages[i])) # downsample
      # stochastic depth probability = 1 - l/L * 0.5
      self.layers.append(ConvNextBlock(stages[i], stochastic_depth_prob=(1 - block_index / L * 0.5))) # residual
      block_index += 1
      self.layers.append(ConvNextBlock(stages[i], stochastic_depth_prob=(1 - block_index / L * 0.5))) # residual
      block_index += 1

    self.layers.append(ConvNextClassifier(stages[-1], out_channels)) # classifier

    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, 0, std=0.02)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, LayerNorm2d):
        nn.init.constant_(m.layer_norm.weight, 1)
        nn.init.constant_(m.layer_norm.bias, 0)

  def forward(self,x):
    #TODO
    for i in range(len(self.layers)):
      x = self.layers[i](x)
    return x
