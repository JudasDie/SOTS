''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: neck modules for SOT models
Data: 2021.6.23
'''
import math
import torch.nn as nn


class ShrinkChannel(nn.Module):
    '''
    shrink feature channel
    '''
    def __init__(self, in_channels, out_channels):
        super(ShrinkChannel, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, crop=False):
        x_ori = self.downsample(x)
        if x_ori.size(3) < 20 and crop:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]

        if not crop:
            return {'ori': x_ori}
        else:
            return {'ori': x_ori, 'crop': xf}


class ShrinkChannelS3S4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShrinkChannelS3S4, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

        self.downsample_s3 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, xs4, xs3):
        xs4 = self.downsample(xs4)
        xs3 = self.downsample_s3(xs3)

        return xs4, xs3


# SiamCAR Neck
class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            if x.size(3) % 2 == 0:
                # return x
                l = 4
                r = l + 8
            else:
                l = 4
                r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=[256, 256, 256]):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]).contiguous())
            return out

class AdjustAllLayer_VLT(nn.Module):
    def __init__(self, backbone_same=False, in_channels=[512, 1024, 2048], out_channels=[512, 1024, 2048]):
        super(AdjustAllLayer_VLT, self).__init__()
        self.backbone_same = backbone_same
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, features, template=False):
        if features[-1].size(-1) < 20:
            template = True
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            len = self.num if self.backbone_same else self.num // 2
            if self.backbone_same or template:
                for i in range(len):
                    adj_layer = getattr(self, 'downsample'+str(i+2))
                    out.append(adj_layer(features[i]).contiguous())
            else:
                for i in range(len):
                    adj_layer = getattr(self, 'downsample' + str(i + 2+len))
                    out.append(adj_layer(features[i]).contiguous())
            return out
