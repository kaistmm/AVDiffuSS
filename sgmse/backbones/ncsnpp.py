# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

import pytorch_lightning as pl

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
from sgmse.util.other import pad_spec

from inspect import isfunction
import math
from torch.nn import functional as F
from torch import einsum
from einops import rearrange, repeat

from .shared import BackboneRegistry
from .attention import SpatialTransformer

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

@BackboneRegistry.register("ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, 
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 1,
        attn_resolutions = (0,), 
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        input_channels = 4,
        spatial_channels = 1,
        dropout = .0,
        centered = False,
        discriminative = False,
        **kwargs):
        super().__init__()

        self.FORCE_STFT_OUT = False
        self.act = act = get_act(nonlinearity)

        self.nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
        
        self.discriminative = discriminative
        if self.discriminative:
            conditional = False
            scale_by_sigma = False
            print("Running NCSN++ as discriminative backbone")
            input_channels = 2  # y.real, y.imag

        self.conditional = conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        self.input_channels = input_channels
        self.spatial_channels = spatial_channels
        self.total_channels = self.input_channels * self.spatial_channels

        self.output_layer = nn.Conv2d(self.total_channels, 2*self.spatial_channels, 1)

        modules = []

        #######################
        ### MODULES NATURES ###
        #######################

        AttnBlock = functools.partial(layerspp.AttnBlockpp, 
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample, 
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir, 
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, 
                dropout=dropout, init_scale=init_scale, 
                skip_rescale=skip_rescale, temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel, 
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        ######################
        ### TIME EMBEDDING ###
        ######################

        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        ##########################
        ### Downsampling block ###
        ##########################

        if progressive_input != 'none':
            input_pyramid_ch = self.total_channels

        modules.append(conv3x3(self.total_channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        
        ##########################
        ### Upsampling block ###
        ##########################

        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), 
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))
                        pyramid_ch = self.total_channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.total_channels, bias=True, init_scale=init_scale))
                        pyramid_ch = self.total_channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    @staticmethod
    def add_argparse_args(parser):
        return parser

    def forward(self, x,context=None, time_cond=None):
        """
        - x: b,2*D,F,T: contains x and y OR x: b,D,F,T contains only x
        """
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts into channel dimensions
        x_chans = []
        for chan in range(self.spatial_channels):
            x_chans.append(torch.cat([ 
                torch.cat([x[:,[chan+in_chan],:,:].real, x[:,[chan+in_chan],:,:].imag ], dim=1) for in_chan in range(self.input_channels // 2)],
                    dim=1)
                )
        x = torch.cat(x_chans, dim=1) #4*D

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas)) if used_sigmas is not None else None
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]  # Input layer: Conv2d
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # edit: check H dim (-2) not W dim (-1)
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:  # Downsampling
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block 
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # Convert to complex number
        h = self.output_layer(h)
        h = torch.reshape(h, (h.size(0), 2, self.spatial_channels, h.size(2), h.size(3)))
        h = torch.permute(h, (0, 2, 3, 4, 1)).contiguous() 
        h = torch.view_as_complex(h)
        return h






class ResNetLayer(pl.LightningModule):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super().__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class GlobalLayerNorm(pl.LightningModule):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

class visualFrontend(pl.LightningModule):
    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    Adopted from TalkNet (https://github.com/TaoRuijie/TalkNet-ASD/tree/main)
    """

    def __init__(self, context_dim=64):
        super().__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, context_dim, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(context_dim, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet(context_dim)
        
        self.visualTCN       = visualTCN(context_dim)
        self.visualConv1D    = visualConv1D(context_dim)
        self.context_dim = context_dim
        return


    def forward(self, inputBatch):
        if inputBatch.ndim!=4:
            B=1
            T, W, H = inputBatch.shape
            inputBatch = inputBatch.view(T, 1, 1, W, H)

        else:
            B, T, W, H = inputBatch.shape
            inputBatch = inputBatch.view(B*T, 1, 1, W, H)

        inputBatch = (inputBatch / 255 - 0.4161) / 0.1688

        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)

        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        
        x = outputBatch.view(B, T, self.context_dim*2)
        
        x = x.transpose(1,2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        
        return x

class ResNet(pl.LightningModule):
    def __init__(self, context_dim=64):
        super().__init__()
        self.context_dim = context_dim
        self.layer1 = ResNetLayer(context_dim, context_dim, stride=1)
        self.layer2 = ResNetLayer(context_dim, context_dim, stride=2)
        self.layer3 = ResNetLayer(context_dim, context_dim*2, stride=2)
        self.layer4 = ResNetLayer(context_dim*2, context_dim*2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch



class DSConv1d(pl.LightningModule):
    def __init__(self, context_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(context_dim*2),
            nn.Conv1d(context_dim*2, context_dim*2, 3, stride=1, padding=1, dilation=1, groups=context_dim*2, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(context_dim*2),
            nn.Conv1d(context_dim*2, context_dim*2, 1, bias=False),
            )

    def forward(self, x):
        out = self.net(x)
        return out + x

class visualTCN(pl.LightningModule):
    def __init__(self, context_dim=64):
        super().__init__()
        stacks = []        
        for x in range(5):
            stacks += [DSConv1d(context_dim)]
        self.net = nn.Sequential(*stacks) # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out

class visualConv1D(pl.LightningModule):
    def __init__(self, context_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(context_dim*2, context_dim, 5, stride=1, padding=2),
            nn.BatchNorm1d(context_dim),
            nn.ReLU(),
            )

    def forward(self, x):
        out = self.net(x)
        return out


@BackboneRegistry.register("ncsnpp_crossatt")
class NCSNpp_crossatt(nn.Module):
    """NCSN++ model with Cross Attention"""
    def __init__(self, 
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 1,
        cross_attn_resolutions = (128,64,32),
        num_heads=4,
        dim_head=16,
        transformer_depth = 1,
        context_dim = 64,
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        input_channels = 4,
        spatial_channels = 1,
        dropout = .0,
        centered = False,
        discriminative = False,
        **kwargs):
        super().__init__()

        self.FORCE_STFT_OUT = False
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.cross_attn_resolutions = cross_attn_resolutions
        self.context_dim = context_dim
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)] 
        
        self.discriminative = discriminative
        if self.discriminative:
            conditional = False
            scale_by_sigma = False
            input_channels = 2 

        self.conditional = conditional 
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        self.input_channels = input_channels
        self.spatial_channels = spatial_channels
        self.total_channels = self.input_channels * self.spatial_channels

        self.output_layer = nn.Conv2d(self.total_channels, 2*self.spatial_channels, 1)

        modules = []

        #######################
        ### MODULES NATURES ###
        #######################

        AttnBlock = functools.partial(layerspp.AttnBlockpp, 
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample, 
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir, 
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, 
                dropout=dropout, init_scale=init_scale, 
                skip_rescale=skip_rescale, temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel, 
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        ######################
        ### TIME EMBEDDING ###
        ######################

        if embedding_type == 'fourier':
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        ##########################
        ### Downsampling block ###
        ##########################

        if progressive_input != 'none':
            input_pyramid_ch = self.total_channels

        modules.append(conv3x3(self.total_channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in cross_attn_resolutions:
                    modules.append(SpatialTransformer(
                        in_ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, res=all_resolutions[i_level]
                        ))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        
        ##########################
        ### Upsampling block ###
        ##########################

        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in cross_attn_resolutions:
                modules.append(SpatialTransformer(
                            in_ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, res=all_resolutions[i_level]
                        ))
                

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), 
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))
                        pyramid_ch = self.total_channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.total_channels, bias=True, init_scale=init_scale))
                        pyramid_ch = self.total_channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.visual_encoder = visualFrontend(context_dim=64)
    @staticmethod
    def add_argparse_args(parser):
        return parser

    def forward(self, x, context=None, time_cond=None):
        context = self.visual_encoder(context)

        modules = self.all_modules
        m_idx = 0

        x_chans = []
        for chan in range(self.spatial_channels):
            x_chans.append(torch.cat([ 
                torch.cat([x[:,[chan+in_chan],:,:].real, x[:,[chan+in_chan],:,:].imag ], dim=1) for in_chan in range(self.input_channels // 2)],
                    dim=1)
                )
            
        x = torch.cat(x_chans, dim=1)

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas)) if used_sigmas is not None else None
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x


        hs = [modules[m_idx](x)]  # Input layer: Conv2d
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                
                if isinstance(modules[m_idx], SpatialTransformer):
                    h = modules[m_idx](h, context)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:  # Downsampling
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block 
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if isinstance(modules[m_idx], SpatialTransformer):
                h = modules[m_idx](h, context)
                m_idx += 1


            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # Convert to complex number
        h = self.output_layer(h) 
        h = torch.reshape(h, (h.size(0), 2, self.spatial_channels, h.size(2), h.size(3)))
        h = torch.permute(h, (0, 2, 3, 4, 1)).contiguous() 
        h = torch.view_as_complex(h) 
        return h, context
