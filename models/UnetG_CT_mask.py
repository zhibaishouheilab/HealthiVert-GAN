"""
基于UNet的生成器，用于输入拼接图像后生成CT和mask
其中下采样共享参数，上采样使用不同的参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    net = UnetGenerator(input_nc, output_nc, 5, ngf, use_dropout=use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)

class DownsampleBlock(nn.Module):
    """Unet下采样模块"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(DownsampleBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpsampleBlock(nn.Module):
    """Unet上采样模块"""
    def __init__(self, in_channels, out_channels, for_mask=False, dropout=0.0):
        super(UpsampleBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.for_mask = for_mask

    def forward(self, x):
        x = self.model(x)
        if self.for_mask:
            x = torch.sigmoid(x)  # 对掩膜使用sigmoid激活函数
        return x

class UnetGenerator(nn.Module):
    """Create a Unet-based generator that generates both CT images and masks"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # Downsample
        self.down_blocks = nn.ModuleList()
        for i in range(num_downs):
            in_channels = input_nc if i == 0 else ngf * 2**(i-1)
            out_channels = ngf * 2**i
            if i == num_downs - 1:  # 最内层不使用标准化
                self.down_blocks.append(DownsampleBlock(in_channels, out_channels, normalize=False, dropout=use_dropout))
            else:
                self.down_blocks.append(DownsampleBlock(in_channels, out_channels, normalize=True, dropout=use_dropout if i>2 else 0.0))
        
        # Upsample for CT and Mask
        self.up_blocks_ct = nn.ModuleList()
        self.up_blocks_mask = nn.ModuleList()
        for i in reversed(range(num_downs)):
            in_channels = ngf * 2**i if i == num_downs - 1 else ngf * 2**(i+1)
            out_channels = ngf * 2**(i-1) if i > 0 else output_nc
            self.up_blocks_ct.append(UpsampleBlock(in_channels, out_channels, for_mask=False, dropout=use_dropout if i<3 else 0.0))
            self.up_blocks_mask.append(UpsampleBlock(in_channels, out_channels, for_mask=(i==0), dropout=use_dropout if i<3 else 0.0))  # Only apply sigmoid in the last layer for mask

    def forward(self, x):
        # Downsample
        features = []
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        
        # Upsample for CT
        x_ct = features[-1]
        for i, block in enumerate(self.up_blocks_ct):
            x_ct = block(x_ct)
            if i < len(features) - 1:  # Skip connection
                x_ct = torch.cat([x_ct, features[-2-i]], dim=1)
        
        # Upsample for Mask
        x_mask = features[-1]
        for i, block in enumerate(self.up_blocks_mask):
            x_mask = block(x_mask)
            if i < len(features) - 1:  # Skip connection
                x_mask = torch.cat([x_mask, features[-2-i]], dim=1)
        
        return x_ct, x_mask
