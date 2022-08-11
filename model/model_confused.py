import math
import numpy as np
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
import copy
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torchvision.models as models
from einops.layers.torch import Rearrange
logger = logging.getLogger(__name__)
import torch.nn.init as init


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(inp_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(inp_dim / 2))
        self.conv2 = Conv(int(inp_dim / 2), int(inp_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(inp_dim / 2))
        self.conv3 = Conv(int(inp_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out



class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)



class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor



class CBamSpatialAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(CBamSpatialAttention,self).__init__()
        kernel_size = 7
        self.att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        out = self._PoolAlongChannel(x)
        out = self.att(out)
        out = torch.sigmoid(out)
        return x*out

    def _PoolAlongChannel(self,x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class CBamChannelAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(CBamChannelAttention,self).__init__()
        self.channel = channel
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(self.channel,self.channel//reduction),
            nn.ReLU(),
            nn.Linear(self.channel//reduction,self.channel)
        )
    def forward(self, x):
        avgPool = F.avg_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
        out1 = self.mlp(avgPool)
        maxPool = F.max_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
        out2 = self.mlp(maxPool)
        out = out1 + out2
        att = torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x*att


# Attention module
class SE_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = torch.sigmoid(y)
        return x*y.expand_as(x)

class CBAM_Attention_Layer(nn.Module):
    def __init__(self, channel,att = 'both', reduction=16):
        super(CBAM_Attention_Layer, self).__init__()
        self.att = att
        self.channelAtt = None
        self.spatialAtt = None
        if att == 'both' or att == 'c':
            self.channelAtt = CBamChannelAttention(channel,reduction)
        if att == 'both' or att == 's':
            self.spatialAtt = CBamSpatialAttention(channel,reduction)



    def forward(self, x):
        if self.att =='both':
            y = self.channelAtt(x)
            y = self.spatialAtt(y)
        elif self.att =='c':
            y = self.channelAtt(x)
        elif self.att =='s':
            y = self.spatialAtt(x)
        return y



class BiFusion(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion, self).__init__()


        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)


        self.residual = Residual(ch_1 + ch_2, ch_out)
        self.conv_d = nn.Conv2d(ch_1, ch_1, kernel_size=3, dilation=4, stride=1, padding=4)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.transbranch = CBAM_Attention_Layer(ch_2,'c',reduction = r_2)
        self.transbranch1 = CBAM_Attention_Layer(ch_1, 's', reduction=r_2)
        self.cnnbranch = SE_Attention_Layer(ch_1,reduction = r_2)

        self.conv = AttentionConv(ch_1, ch_1, kernel_size=3, padding=1)
        #self.conv = AttentionStem(in_channels=ch_1, out_channels=ch_1, kernel_size=3, stride=1, padding=1, groups=1)
        self.cnnbr = ChannelGate(ch_1)


    def forward(self, g, x):

        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        #g = self.conv_d(g)
        # g = self.cnnbranch(g)
        g = self.transbranch1(g)
        #g = self.cnnbr(g)



        # channel attetion for transformer branch
        x = self.transbranch(x)

        fuse = self.residual(torch.cat([g, x], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse











class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)

















class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops






class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.conv = nn.Conv2d(channel, channel // reduction, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(channel // 2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):

        kaiming_init(self.conv, mode='fan_in')

        self.conv.inited = True

    def forward(self, x):
        s = x
        s = self.gelu(self.bn(self.conv(s)))
        b, c, h, w = s.size()
        avg = self.pool(s)
        avg = avg.view(b, c, 1).permute(0, 2, 1)
        s = s.view(b, c, h * w)
        text = torch.matmul(avg, s)
        text = self.softmax(text)
        text = text.view(b, 1, h, w)
        mask = self.sigmoid(text)

        out = x * mask

        out = self.gelu(self.bn1(self.conv1(out)))
        #out = self.conv1(out)



        return out

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


        self.gelu = nn.GELU()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim // 2)
        self.bn2 = nn.BatchNorm2d(dim)

        self.sim = SELayer(channel=dim)

    def forward(self, x):
        # Trans_feature
        i = 0
        for blk in self.blocks:
            if i % 2 == 0:
                s = x
                H, W = self.input_resolution
                B, L, C = s.shape  # [1, 16384, 96]

                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                s = s.view(B, H, W, C)  # ([1, 128, 128, 96])
                s = s.permute(0, 3, 1, 2).contiguous()

                # s = self.gelu(self.bn1(self.conv1(s)))
                # s_h, s_w = self.pool_h(s), self.pool_w(s)  # .permute(0, 1, 3, 2)
                # s = torch.matmul(s_h, s_w)
                # s = self.gelu(self.bn2(self.conv2(s)))
                s = self.sim(s)


            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

            i=i+1
            if i % 2 == 0:
                H, W = self.input_resolution
                B, L, C = x.shape  # ([1, 16384, 96])
                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                x = x.view(B, H, W, C)  # ([12, 64, 64, 384])
                x = x.permute(0, 3, 1, 2).contiguous()
                x=x+s
                x = x.view(B, -1, C)

        if self.downsample is not None:
            x = self.downsample(x)
        return x




    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops






class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchMerging1(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W , "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, 2 * C)
        return x





class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        self.ch_1 = ch_1

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(ch_2, ch_2 // 2),
            nn.ReLU()
            #nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.incr = nn.Linear(ch_2 // 2, ch_2)
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, g, x):             #CNN、TRANS
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch   spatial
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch   senet
        x_in = x
        #x = x.mean((2, 3), keepdim=True)
        s_h, s_w = self.pool_h(x), self.pool_w(x)  # .permute(0, 1, 3, 2)
        x = torch.matmul(s_h, s_w)


        # max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # maxpoolmlp = self.mlp(max_pool)
        # channel_att_sum = self.incr(maxpoolmlp)
        # x = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)



        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.sigmoid(x) * x_in
        #
        # avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # avgpoolmlp = self.mlp(avg_pool)
        # maxpoolmlp=self.mlp(max_pool)
        # pooladd = avgpoolmlp+maxpoolmlp
        # channel_att_sum = self.incr(pooladd)
        # x = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # x = x * x_in


        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse






class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = DecoderBlock(in_channels=256,out_channels=256,skip_channels = 256)
        self.block1 = DecoderBlock(in_channels=256, out_channels=64,skip_channels = 64)

    def forward(self,features,x0):
        x = features[0]
        for i in range(3):
            skip = features[i+1]
            x = self.block(x, skip)
        x = self.block1(x, x0)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels,
            use_batchnorm=True,
    ):
        super().__init__()

        self.conv1 = Conv(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            bn = use_batchnorm
        )
        self.conv2 = Conv(
            out_channels,
            out_channels,
            kernel_size=1,
            bn = use_batchnorm
        )
        #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conT=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, skip):
        x = self.up(x)
        #print(x.shape, skip.shape)

        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)

        x = self.conv2(x)

        return x









class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            #nn.AvgPool2d(2)
            #nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Encoder_patch(nn.Module):
    def __init__(self, n_channels, emb_size=96):
        super(Encoder_patch, self).__init__()
        self.n_channels = n_channels
        self.emb_size = emb_size

        self.conv1 = DoubleConv(n_channels, 64)
        self.conv2 = DoubleConv(64, emb_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), start_dim=1)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels: int = 3, patch_shape: int = 4, emb_size: int = 96, img_shape: int = 224, norm_layer = None):
        img_size = to_2tuple(img_shape)
        patch_size = to_2tuple(patch_shape)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patch_size = patch_size
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_channels
        self.embed_dim = emb_size
        super().__init__()
        #         self.encoder = Encoder_patch(n_channels = in_channels, emb_size= emb_size)
        self.projection = nn.Sequential(
            Rearrange('b c (ph h) (pw w) -> b c (ph pw) h w', c=in_channels, h=patch_shape, ph=img_shape // patch_shape,
                      w=patch_shape, pw=img_shape // patch_shape),
            Rearrange('b c p h w -> (b p) c h w'),
            Encoder_patch(n_channels=in_channels, emb_size=emb_size),
            Rearrange('(b p) d-> b p d', p=(img_shape // patch_shape) ** 2),
        )
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        if self.norm is not None:
            x = self.norm(x)
        return x





class swin_fusion(nn.Module):
    def __init__(self, img_size=768, patch_size=4, in_chans=3, num_classes=19,
                 embed_dim=128, depths=[2, 2, 18, 2], depths_decoder=[1, 2, 2, 2], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=True,**kwargs):
        super().__init__()

        self.resnet = models.resnet101(pretrained=True)
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.num_layers = len(depths)
        self.initial = Conv(inp_dim=3,out_dim=32,bn=True)
        #self.initial = AttentionConv(3, 32, kernel_size=3, padding=1)

        self.last = nn.Sequential(OrderedDict([
            ('conv33', nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)),
            ('norm33_0', nn.BatchNorm2d(96)),
            ('relu33_0', nn.ReLU(inplace=True)),
            ('conv11' , nn.Conv2d(96, num_classes, kernel_size=1))
        ]))


       # split image into non-overlapping patches
       #  self.patch_embed = PatchEmbed(
       #      img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
       #      norm_layer=norm_layer if patch_norm else None)
       #  num_patches = self.patch_embed.num_patches
       #  patches_resolution = self.patch_embed.patches_resolution
       #  self.patches_resolution = patches_resolution

        #patch_embedding
        self.patch_embed = PatchEmbedding(in_channels =in_chans , patch_shape =patch_size ,
                                             emb_size = embed_dim, img_shape = img_size, norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        outchannel = [256,512,1024,2048]   #[256,512,1024,2048]         #[64,128,256,512]
        r_2 = [1,2,2,4]
        self.layers = nn.ModuleList()
        self.Down_features = nn.ModuleList()
        self.Fuse = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            downsample = PatchMerging1(input_resolution=layer.input_resolution, dim=layer.dim, norm_layer=norm_layer)
            self.layers.append(layer)
            fuse = BiFusion_block(ch_1=outchannel[i_layer], ch_2=layer.dim, r_2=r_2[i_layer],
                                       ch_int=256, ch_out=256, drop_rate=drop_rate)
            self.Down_features.append(downsample)
            self.Fuse.append(fuse)

        self.decoder = DecoderCup()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        Trans_features = []
        input = x
        initial =self.initial(input)

        x = self.patch_embed(input)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        trans_x = x
        i_layer = 0
        self.num_layers = len(self.layers)
        for layer in self.layers:
            trans_x = layer(trans_x)
            B, L, C = trans_x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(L)), int(np.sqrt(L))
            Trans_x = trans_x.permute(0, 2, 1)        #B C HW
            Trans_x = Trans_x.contiguous().view(B, C, h, w)
            Trans_features.append(Trans_x)

            if (i_layer < self.num_layers - 1):
                trans_x = self.Down_features[i_layer](trans_x)
            i_layer = i_layer + 1


        #resnet
        x0 = self.resnet.conv1(input)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        c0 = x0
        x0 = self.resnet.maxpool(x0)
        c1 = self.resnet.layer1(x0)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        c4 = self.resnet.layer4(c3)
        features = [c1, c2, c3, c4]



        # for i in range(4):
        #     print(Trans_features[i].shape,features[i].shape)

        #特征融合
        fuse_features = []
        for i,fuse in enumerate(self.Fuse):
            fuse_feature= fuse(features[i],Trans_features[i])
            fuse_features.append(fuse_feature)
        fuse_features = fuse_features[::-1]


        #上采样
        x = self.decoder(fuse_features,c0)

        x = self.upsampling(x)

        x = torch.cat([x, initial], dim=1)
        x = self.last(x)
        return x
