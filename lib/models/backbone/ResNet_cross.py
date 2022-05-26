import math

import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, used_layers=[2,3,4], cross_layers=[1, 2, 3, 4], self_attn_plus=True,
                 stage_dims=[256,512,1024,2048], map_dims=[64,128,256,512], num_heads=4, group=8):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # cross
        self.self_attn_plus = self_attn_plus
        self.cross_layers = cross_layers
        self.num_heads = num_heads
        self.group = group
        if cross_layers is not None:
            ks = 3
            pd = 1
            bs = True
            self.cross_map_q = nn.ModuleList([
                nn.Conv2d(stage_dims[0], map_dims[0], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[1], map_dims[1], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[2], map_dims[2], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[3], map_dims[3], kernel_size=ks, padding=pd, bias=bs),
            ])
            self.cross_map_k = nn.ModuleList([
                nn.Conv2d(stage_dims[0], map_dims[0], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[1], map_dims[1], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[2], map_dims[2], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[3], map_dims[3], kernel_size=ks, padding=pd, bias=bs),
            ])
            self.cross_map_v = nn.ModuleList([
                nn.Conv2d(stage_dims[0], map_dims[0], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[1], map_dims[1], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[2], map_dims[2], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(stage_dims[3], map_dims[3], kernel_size=ks, padding=pd, bias=bs),
            ])
            self.cross_proj = nn.ModuleList([
                nn.Conv2d(map_dims[0], stage_dims[0], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(map_dims[1], stage_dims[1], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(map_dims[2], stage_dims[2], kernel_size=ks, padding=pd, bias=bs),
                nn.Conv2d(map_dims[3], stage_dims[3], kernel_size=ks, padding=pd, bias=bs),
            ])
            if self_attn_plus:
                self.self_map_q = nn.ModuleList([
                    nn.Conv2d(stage_dims[0], map_dims[0], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[1], map_dims[1], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[2], map_dims[2], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[3], map_dims[3], kernel_size=ks, padding=pd, bias=bs),
                ])
                self.self_map_k = nn.ModuleList([
                    nn.Conv2d(stage_dims[0], map_dims[0], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[1], map_dims[1], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[2], map_dims[2], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[3], map_dims[3], kernel_size=ks, padding=pd, bias=bs),
                ])
                self.self_map_v = nn.ModuleList([
                    nn.Conv2d(stage_dims[0], map_dims[0], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[1], map_dims[1], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[2], map_dims[2], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(stage_dims[3], map_dims[3], kernel_size=ks, padding=pd, bias=bs),
                ])
                self.self_proj = nn.ModuleList([
                    nn.Conv2d(map_dims[0], stage_dims[0], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(map_dims[1], stage_dims[1], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(map_dims[2], stage_dims[2], kernel_size=ks, padding=pd, bias=bs),
                    nn.Conv2d(map_dims[3], stage_dims[3], kernel_size=ks, padding=pd, bias=bs),
                ])
            self.proj_drop = nn.Dropout(0.1)
            self.softmax = nn.Softmax(dim=-1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,  # kernel=3
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.size(-1) < 200:
            template = True
        else:
            template = False
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        if not template and 1 in self.cross_layers:
            p1 = self.cross_attn(p1, self.templates[1], id=1 - self.cross_layers[0], num_heads=self.num_heads, group=self.group)
            if self.self_attn_plus:
                p1 = self.self_attn(p1, p1, id=1 - self.cross_layers[0])
        p2 = self.layer2(p1)
        if not template and 2 in self.cross_layers:
            p2 = self.cross_attn(p2, self.templates[2], id=2 - self.cross_layers[0], num_heads=self.num_heads, group=self.group)
            if self.self_attn_plus:
                p2 = self.self_attn(p2, p2, id=2 - self.cross_layers[0])
        p3 = self.layer3(p2)
        if not template and 3 in self.cross_layers:
            p3 = self.cross_attn(p3, self.templates[3], id=3 - self.cross_layers[0], num_heads=self.num_heads, group=self.group)
            if self.self_attn_plus:
                p3 = self.self_attn(p3, p3, id=3 - self.cross_layers[0])
        p4 = self.layer4(p3)
        if not template and 4 in self.cross_layers:
            p4 = self.cross_attn(p4, self.templates[4], id=4 - self.cross_layers[0], num_heads=self.num_heads, group=self.group)
            if self.self_attn_plus:
                p4 = self.self_attn(p4, p4, id=4 - self.cross_layers[0])
        out = [x_, p1, p2, p3, p4]
        if template:
            self.templates = out
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def cross_attn(self, x, z, num_heads=4, group=8, id=0):
        batchsize, C, H_x, W_x = x.shape
        C = self.cross_proj[id].weight.size(1)
        scale_factor = (C / num_heads) ** -0.5
        H_z, W_z = z.shape[-2:]
        if group > 0:
            x_pad, H_x, W_x, Hp_x, Wp_x = self.cross_pad(x, group)
            z_pad, H_z, W_z, Hp_z, Wp_z = self.cross_pad(z, group)

            num_win_hw = int(Hp_x // group)
            num_win_hw_template = int(Hp_z // group)
            x_cross = self.cross_map_q[id](x_pad).reshape(batchsize, num_heads, -1, num_win_hw, group, num_win_hw,
                                                          group).permute(0, 1, 4, 6, 3, 5, 2).reshape(batchsize,
                                                                                                      num_heads,
                                                                                                      group * group,
                                                                                                      num_win_hw * num_win_hw,
                                                                                                      -1)
            template_k = self.cross_map_k[id](z_pad).reshape(batchsize, num_heads, -1, num_win_hw_template, group,
                                                             num_win_hw_template, group).permute(0, 1, 4, 6, 3, 5,
                                                                                                 2).reshape(batchsize,
                                                                                                            num_heads,
                                                                                                            group * group,
                                                                                                            num_win_hw_template * num_win_hw_template,
                                                                                                            -1)
            template_v = self.cross_map_v[id](z_pad).reshape(batchsize, num_heads, -1, num_win_hw_template, group,
                                                             num_win_hw_template, group).permute(0, 1, 4, 6, 3, 5,
                                                                                                 2).reshape(batchsize,
                                                                                                            num_heads,
                                                                                                            group * group,
                                                                                                            num_win_hw_template * num_win_hw_template,
                                                                                                            -1)
        else:
            x_cross = self.cross_map_q[id](x).reshape(batchsize, num_heads, -1, H_x * W_x).permute(0, 1, 3, 2)
            template_k = self.cross_map_k[id](z).reshape(batchsize, num_heads, -1, H_z * W_z).permute(0, 1, 3, 2)
            template_v = self.cross_map_v[id](z).reshape(batchsize, num_heads, -1, H_z * W_z).permute(0, 1, 3, 2)

        x_cross_scaled = self.softmax(x_cross * scale_factor)
        weight_matrix = x_cross_scaled @ template_k.transpose(-1, -2)
        if group > 0:
            x_cross = (weight_matrix @ template_v).permute(0, 1, 4, 3, 2).reshape(batchsize, num_heads, -1, num_win_hw,
                                                                                  num_win_hw, group, group).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    3,
                                                                                                                    5,
                                                                                                                    4,
                                                                                                                    6).reshape(
                batchsize, C, Hp_x, Wp_x)
            if H_x != Hp_x or W_x != Wp_x:
                x_cross = x_cross[:, :, :H_x, :W_x].contiguous()
        else:
            x_cross = (weight_matrix @ template_v).transpose(-1, -2).reshape(batchsize, C, H_x, W_x)
        x = x + self.proj_drop(self.cross_proj[id](x_cross))
        return x

    def self_attn(self, x, z, num_heads=4, group=8, id=0):
        batchsize, C, H_x, W_x = x.shape
        C = self.self_proj[id].weight.size(1)
        scale_factor = (C / num_heads) ** -0.5
        H_z, W_z = z.shape[-2:]
        if group > 0:
            x_pad, H_x, W_x, Hp_x, Wp_x = self.cross_pad(x, group)
            z_pad, H_z, W_z, Hp_z, Wp_z = self.cross_pad(z, group)

            num_win_hw = int(Hp_x // group)
            num_win_hw_template = int(Hp_z // group)
            x_cross = self.self_map_q[id](x_pad).reshape(batchsize, num_heads, -1, num_win_hw, group, num_win_hw,
                                                          group).permute(0, 1, 4, 6, 3, 5, 2).reshape(batchsize,
                                                                                                      num_heads,
                                                                                                      group * group,
                                                                                                      num_win_hw * num_win_hw,
                                                                                                      -1)
            template_k = self.self_map_k[id](z_pad).reshape(batchsize, num_heads, -1, num_win_hw_template, group,
                                                             num_win_hw_template, group).permute(0, 1, 4, 6, 3, 5,
                                                                                                 2).reshape(batchsize,
                                                                                                            num_heads,
                                                                                                            group * group,
                                                                                                            num_win_hw_template * num_win_hw_template,
                                                                                                            -1)
            template_v = self.self_map_v[id](z_pad).reshape(batchsize, num_heads, -1, num_win_hw_template, group,
                                                             num_win_hw_template, group).permute(0, 1, 4, 6, 3, 5,
                                                                                                 2).reshape(batchsize,
                                                                                                            num_heads,
                                                                                                            group * group,
                                                                                                            num_win_hw_template * num_win_hw_template,
                                                                                                            -1)
        else:
            x_cross = self.self_map_q[id](x).reshape(batchsize, num_heads, -1, H_x * W_x).permute(0, 1, 3, 2)
            template_k = self.self_map_k[id](z).reshape(batchsize, num_heads, -1, H_z * W_z).permute(0, 1, 3, 2)
            template_v = self.self_map_v[id](z).reshape(batchsize, num_heads, -1, H_z * W_z).permute(0, 1, 3, 2)

        x_cross_scaled = self.softmax(x_cross * scale_factor)
        weight_matrix = x_cross_scaled @ template_k.transpose(-1, -2)
        if group > 0:
            x_cross = (weight_matrix @ template_v).permute(0, 1, 4, 3, 2).reshape(batchsize, num_heads, -1, num_win_hw,
                                                                                  num_win_hw, group, group).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    3,
                                                                                                                    5,
                                                                                                                    4,
                                                                                                                    6).reshape(
                batchsize, C, Hp_x, Wp_x)
            if H_x != Hp_x or W_x != Wp_x:
                x_cross = x_cross[:, :, :H_x, :W_x].contiguous()
        else:
            x_cross = (weight_matrix @ template_v).transpose(-1, -2).reshape(batchsize, C, H_x, W_x)
        x = x + self.proj_drop(self.self_proj[id](x_cross))
        return x

    def cross_pad(self, x, group):
        H, W = x.shape[-2:]
        pad_l = pad_t = 0
        pad_r = (group - W % group) % group
        pad_b = (group - H % group) % group
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape
        return x, H, W, Hp, Wp


def ResNet50_cross(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained = True
    if pretrained:
        check = model_zoo.load_url(model_urls['resnet50'])
        new_check = dict()
        for key in list(check.keys()):
            if key == 'layer2.0.downsample.0.weight' or key == 'layer3.0.downsample.0.weight' or key == 'layer4.0.downsample.0.weight':
                continue
            else:
                new_check[key] = check[key]
        model.load_state_dict(new_check, strict=False)
    return model


if __name__ == '__main__':
    net = ResNet50_cross(used_layers=[2, 3, 4])
    # print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()
    # var = Variable(var)

    net(var)
    print('*************')
    var = torch.FloatTensor(1, 3, 255, 255).cuda()
    # var = Variable(var)
    net(var)
