import torch
import math
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.functional.lif import lif_feed_forward_step, LIFFeedForwardState, LIFParameters

# https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

class ConvLSTMCellSpike(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, seq_length=100):
        super(ConvLSTMCellSpike, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = Parameter(torch.Tensor(
            4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(
            4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = Parameter(torch.Tensor(
            3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))
        
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.seq_length = seq_length
        self.lif_parameters = LIFParameters(method="super", alpha=torch.tensor(100))
        self.lif_t_parameters = LIFParameters(method="tanh", alpha=torch.tensor(100))

        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                    self.padding_h, self.dilation, self.groups)

        # Cell uses a Hadamard product instead of a convolution?
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                    self.padding_h, self.dilation, self.groups)

        xs = self.constant_current_encoder(input)
        
        si = sf = sg = so = LIFFeedForwardState(0, 0) # v, i = 0
        vi = []
        vf = []
        vg = []
        vo = []
        for x in xs:
            wx = F.conv2d(x, self.weight_ih, self.bias_ih,
                        self.stride, self.padding, self.dilation, self.groups)

            wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(
                wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)
            
            _, si = lif_feed_forward_step(wxhc[:, :self.out_channels], si, self.lif_parameters)
            _, sf = lif_feed_forward_step(wxhc[:, self.out_channels:2 * self.out_channels], sf)
            _, sg = lif_feed_forward_step(wxhc[:, 2 * self.out_channels:3 * self.out_channels], sg, self.lif_t_parameters)
            _, so = lif_feed_forward_step(wxhc[:, 3 * self.out_channels:], sg, self.lif_parameters)
            vi.append(si.v)
            vf.append(sf.v)
            vg.append(sg.v)
            vo.append(so.v)
        i = torch.stack(vi[1:]).max(0).values
        f = torch.stack(vf[1:]).max(0).values
        g = torch.stack(vg[1:]).max(0).values
        o = torch.stack(vo[1:]).max(0).values

        c_1 = f * c_0 + i * g
        h_1 = o * F.tanh(c_1)
        return h_1, (h_1, c_1)
