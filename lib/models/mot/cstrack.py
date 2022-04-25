''' Details
Author: Zhipeng Zhang/Chao Liang
Function: build mot models
Data: 2022.4.7
'''

import argparse
import math
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from easydict import EasyDict as edict

from utils.general_helper import make_divisible, time_synchronized
from utils.read_file import check_file
from utils.log_helper import set_logging
from utils.model_helper import model_info, initialize_weights, select_device
from utils.tracking_helper import scale_img

from models.mot.detector.YOLOv5 import Detect, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, \
                                       Concat, ASPP, _NonLocalBlockND, MixConv2d, CrossConv, C3, check_anchor_order, \
                                        fuse_conv_and_bn
from loguru import logger
# logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict


        self.yaml = edict(self.yaml)
        # Define model
        if nc and nc != self.yaml.MODEL.nc:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml.MODEL.nc = nc
        self.model, self.save, self.out = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out



        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            x = self.forward(torch.zeros(2, ch, s, s))
            m.stride = torch.tensor([s / x.shape[-2] for x in x[-1][0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        output = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            if m.i in self.out:
                output.append(x)
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return output

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    detector_type = d.MODEL.DetectorType
    detector_cfg = d.MODEL.Detector[detector_type]

    anchors, nc, gd, gw, id_embedding = detector_cfg.anchors, d.MODEL.nc, detector_cfg.depth_multiple, \
                                        detector_cfg.width_multiple, d.MODEL.id_embedding
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    out_list = []
    for i, (f, n, m, args) in enumerate(detector_cfg.backbone + detector_cfg.head):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            if i == 24:
                args = [c2, c1, *args[1:]]
            else:
                args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            out_list += [i]
        elif m is SAAN:
            out_list += [i]
            args.append([ch[x + 1] for x in f])
        elif m is DenseMask:
            out_list += [i]
            args.append([ch[x + 1] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), sorted(out_list)


class DenseMask(nn.Module):
    """
    predict pedestrian density, used for NMS
    """

    def __init__(self, mask=1, ch=()):
        super(DenseMask, self).__init__()
        self.proj1 = Conv(ch[0]//2, 1,k=3)
        self.proj2 = nn.ConvTranspose2d(ch[1], 1, 4, stride=2,
                                                     padding=1, output_padding=0,
                                                     groups=1, bias=False)
        self.proj3 = nn.ConvTranspose2d(ch[2], 1, 8, stride=4,
                                                     padding=2, output_padding=0,
                                                     groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, layers):
        return self.sigmoid(self.proj1(layers[0][0])+self.proj2(layers[1][0])+self.proj3(layers[2][0]))


class SAAN(nn.Module):
    """
    SAAN modules proposed in CSTrack
    """
    def __init__(self, id_embedding=256, ch=()):
        super(SAAN, self).__init__()
        self.proj1 = nn.Sequential(Conv(ch[0]//2, 256, k=3),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))
        self.proj2 = nn.Sequential(Conv(ch[1], 256, k=3),
                                    nn.ConvTranspose2d(256, 256, 4, stride=2,
                                                       padding=1, output_padding=0,
                                                       groups=256, bias=False),
                                    SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))
        self.proj3 = nn.Sequential(Conv(ch[2], 256, k=3),
                                   nn.ConvTranspose2d(256, 256, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=256, bias=False),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))

        self.node = nn.Sequential(SAAN_Attention(k_size=3, ch=256*3, s_state=False, c_state=True),
                                  Conv(256 * 3, 256,k=3),
                                  nn.Conv2d(256, id_embedding,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=True)
                                  )

    def forward(self, layers):
        layers[0] = self.proj1(layers[0][1])
        layers[1] = self.proj2(layers[1][1])
        layers[2] = self.proj3(layers[2][1])

        id_layer_out = self.node(torch.cat([layers[0], layers[1], layers[2]], 1))
        id_layer_out = id_layer_out.permute(0, 2, 3, 1).contiguous()
        return id_layer_out


class SAAN_Attention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3, ch=256, s_state=False, c_state=False):
        super(SAAN_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.s_state = s_state
        self.c_state = c_state

        if c_state:
            self.c_attention = nn.Sequential(nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                                                  nn.LayerNorm([1, ch]),
                                                  nn.LeakyReLU(0.3, inplace=True),
                                                  nn.Linear(ch, ch, bias=False))

        if s_state:
            self.conv_s = nn.Sequential(Conv(ch, ch // 4, k=1))
            self.s_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # channel_attention
        if self.c_state:
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_c = self.c_attention(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)+\
                  self.c_attention(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y_c = self.sigmoid(y_c)

        #spatial_attention
        if self.s_state:
            x_s = self.conv_s(x)
            avg_out = torch.mean(x_s, dim=1, keepdim=True)
            max_out, _ = torch.max(x_s, dim=1, keepdim=True)
            y_s = torch.cat([avg_out, max_out], dim=1)
            y_s = self.sigmoid(self.s_attention(y_s))

        if self.c_state and self.s_state:
            y = x * y_s * y_c + x
        elif self.c_state:
            y = x * y_c + x
        elif self.s_state:
            y = x * y_s + x
        else:
            y = x
        return y


class CCN(nn.Module):
    def __init__(self, k_size=3, ch=()):
        super(CCN, self).__init__()
        self.w1 = Parameter(torch.ones(1)*0.5)
        self.w2 = Parameter(torch.ones(1)*0.5)
        w, h = 6, 10

        self.avg_pool = nn.AdaptiveAvgPool2d((w, h))

        self.c_attention1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.InstanceNorm2d(num_features=ch),
                                          nn.LeakyReLU(0.3, inplace=True))
        self.c_attention2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.InstanceNorm2d(num_features=ch),
                                          nn.LeakyReLU(0.3, inplace=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y_t1 = self.c_attention1(y)
        y_t2 = self.c_attention2(y)
        bs, c, h, w = y_t1.shape
        y_t1 =y_t1.view(bs, c, h*w)
        y_t2 =y_t2.view(bs, c, h*w)

        y_t1_T = y_t1.permute(0, 2, 1)
        y_t2_T = y_t2.permute(0, 2, 1)
        M_t1 = torch.matmul(y_t1, y_t1_T)
        M_t2 = torch.matmul(y_t2, y_t2_T)
        M_t1 = F.softmax(M_t1, dim=-1)
        M_t2 = F.softmax(M_t2, dim=-1)

        M_s1 = torch.matmul(y_t1, y_t2_T)
        M_s2 = torch.matmul(y_t2, y_t1_T)
        M_s1 = F.softmax(M_s1, dim=-1)
        M_s2 = F.softmax(M_s2, dim=-1)

        x_t1 = x
        x_t2 = x
        bs, c, h, w = x_t1.shape
        x_t1 = x_t1.contiguous().view(bs, c, h*w)
        x_t2 = x_t2.contiguous().view(bs, c, h*w)

        x_t1 = torch.matmul(self.w1*M_t1 + (1-self.w1)*M_s1, x_t1).contiguous().view(bs, c, h, w)
        x_t2 = torch.matmul(self.w2*M_t2 + (1-self.w2)*M_s2, x_t2).contiguous().view(bs, c, h, w)
        return [x_t1+x, x_t2+x]





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
