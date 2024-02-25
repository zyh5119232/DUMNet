import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
import math

# 重写Conv2d函数，降低计算量
class conv2d_1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3,3),
                 stride=1,
                 padding=2,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(conv2d_1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, int(round(in_channels / groups)), *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        return f.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
