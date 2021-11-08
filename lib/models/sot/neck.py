''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: neck modules for SOT models
Data: 2021.6.23
'''

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