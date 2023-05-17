#  Copyright 2023 Laboratory for Integrative Personalized Medicine (PIMed), Stanford University, USA
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import print_function, division
from torch import nn, cat
from torch.utils.tensorboard import SummaryWriter
from torch import rand
from torchsummaryX import summary
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
        """
        Adapted from the CPVR 2021 paper: https://arxiv.org/abs/2103.02907
        https://github.com/houqb/CoordAttention/blob/main/coordatt.py
        """
    def __init__(self, inp, oup, reduction=32):

        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class Encoder(nn.Module):
    def __init__(
            self, filters=64,
            in_channels=3,
            n_block=3,
            kernel_size=(3, 3),
            batch_norm=True,
            padding='same',
            attention=False
    ):
        super().__init__()
        self.filter = filters
        for i in range(n_block):
            out_ch = filters * 2 ** i
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters * 2 ** (i - 1)

            if padding == 'same':
                pad = kernel_size[0] // 2
            else:
                pad = 0
            model = [nn.Conv2d(in_channels=in_ch, 
                               out_channels=out_ch, 
                               kernel_size=kernel_size, 
                               padding=pad),
                     nn.ReLU(inplace=True)
                    ]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, 
                                out_channels=out_ch, 
                                kernel_size=kernel_size, 
                                padding=pad),
                      nn.ReLU(inplace=True)
                     ]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            
            # you can put attention to false if you don't want in encoder part.
            if attention:
                model += [CoordAtt(inp=out_ch, oup=out_ch)]
            self.add_module('encoder%d' % (i + 1), nn.Sequential(*model))
            conv = [nn.Conv2d(in_channels=in_ch * 3, out_channels=out_ch, kernel_size=1), nn.ReLU(inplace=True)]
            self.add_module('conv1_%d' % (i + 1), nn.Sequential(*conv))

    def forward(self, x):
        skip = []
        output = x
        res = None
        i = 0
        for name, layer in self._modules.items():
            if i % 2 == 0:
                output = layer(output)
                skip.append(output)
            else:
                if i > 1:
                    output = cat([output, res], 1)
                    output = layer(output)
                output = nn.MaxPool2d(kernel_size=(2,2))(output)
                res = output
            i += 1
        return output, skip


class Bottleneck(nn.Module):
    def __init__(self, filters=64, n_block=3, depth=4, kernel_size=(3,3)):
        super().__init__()
        out_ch = filters * 2 ** n_block
        in_ch = filters * 2 ** (n_block - 1)
        for i in range(depth):
            dilate = 2 ** i
            model = [nn.Conv2d(in_channels=in_ch, 
                               out_channels=out_ch, 
                               kernel_size=kernel_size,
                               padding=dilate,
                               dilation=dilate), 
                     nn.ReLU(inplace=True)]
            self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*model))
            if i == 0:
                in_ch = out_ch

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class Decoder(nn.Module):
    def __init__(
            self,
            filters=64,
            n_block=3,
            kernel_size=(3, 3),
            batch_norm=True,
            padding='same',
            attention=False
    ):
        super().__init__()
        self.n_block = n_block
        if padding == 'same':
            pad = kernel_size[0] // 2
        else:
            pad = 0
        for i in reversed(range(n_block)):
            out_ch = filters * 2 ** i
            in_ch = 2 * out_ch
            model = [nn.UpsamplingNearest2d(scale_factor=(2, 2)),
                     nn.Conv2d(in_channels=in_ch, 
                               out_channels=out_ch, 
                               kernel_size=kernel_size,
                               padding=pad)]
            self.add_module('decoder1_%d' % (i + 1), nn.Sequential(*model))

            model = [nn.Conv2d(in_channels=in_ch, 
                               out_channels=out_ch, 
                               kernel_size=kernel_size, 
                               padding=pad),
                     nn.ReLU(inplace=True)
                    ]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, 
                                out_channels=out_ch,
                                kernel_size=kernel_size, 
                                padding=pad),
                      nn.ReLU(inplace=True)
                     ]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            # you can put attention to false if you don't want in encoder part.
            if attention:
                model += [CoordAtt(inp=out_ch, oup=out_ch)]
            self.add_module('decoder2_%d' % (i + 1), nn.Sequential(*model))

    def forward(self, x, skip):
        i = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            if i % 2 == 0:
                output = cat([skip.pop(), output], 1)
            i += 1
        return output


class Segmentation_model(nn.Module):
    def __init__(
            self,
            filters=32,
            in_channels=3,
            n_block=4,
            bottleneck_depth=4,
            n_class=3,
            attention=False
    ):
        super().__init__()
        # encoder class
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block, attention=attention)
        # dialated bottleneck class
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        # decoder class
        self.decoder = Decoder(filters=filters, n_block=n_block, attention=attention)
        # classifier class
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))

    def forward(self, x, features_out=False):
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)
        output = self.decoder(output_bottleneck, skip)
        output = self.classifier(output)
        # if you need the bottleneck feature make features_out to true
        if features_out:
            return output, output_bottleneck
        else:
            return output


if __name__ == '__main__':
    net = Segmentation_model(filters=32, n_block=4, in_channels=1, attention=True).cuda()
    arch = summary(net, torch.rand((1, 1, 128, 160)).cuda())
    print(arch)
