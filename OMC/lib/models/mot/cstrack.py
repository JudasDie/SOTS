
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.mot.common import *
from models.mot.experimental import *
from models.mot.autoanchor import check_anchor_order
from core.mot.general import make_divisible, check_file, set_logging
from core.mot.plots import feature_visualization
from core.mot.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

from core.mot.torch_utils import intersect_dicts
from core.mot.base_trainer import build_targets_siammot
from dataset.cstrack import draw_gaussian_only,gaussian_radius
import numpy as np
from core.mot.general import non_max_suppression_and_inds


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

logger = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), id_embedding=256, ch=(), inplace=False):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList([nn.Conv2d(ch[0] // 2, (self.no) * self.na, 1),
                                nn.Conv2d(ch[1], (self.no) * self.na, 1),
                                nn.Conv2d(ch[2], (self.no) * self.na, 1)])  # output conv

        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.id_embedding = id_embedding
        self.k = Parameter(torch.ones(1) * 10)

    def forward(self, x):

        x_ori = x.copy()  # for profiling
        z = []  # inference output
        p = []
        for i in range(self.nl):
            # print(x[i][0].size())

            x[i] = self.m[i](x[i][0])  # conv
            # x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                p.append(y.clone())
                y[..., 0:2] = ((y[..., 0:2] - 0.5) * self.k + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y = y[..., :6]
                p[-1][..., 2:] = y[..., 2:]
                z.append(y.view(bs, -1, self.no))
            else:
                if self.stride != None:
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    p.append(y.clone())
                    y[..., 0:2] = ((y[..., 0:2] - 0.5) * self.k + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = y[..., :6]
                    z.append(y.view(bs, -1, self.no))
                else:
                    z = [torch.zeros((1,1,1)).to(x[i].device)]*3

        # return x if self.training else (torch.cat(z, 1), x)
        return [x,self.k,x_ori,torch.cat(z, 1)] if self.training else (torch.cat(z, 1), [x,self.k,x_ori,p,self.stride,self.grid,self.anchor_grid,self.no,bs])

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

'''
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), id_embedding=256, ch=(), inplace=False):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList([nn.Conv2d(ch[0] // 2, (self.no) * self.na, 1),
                                nn.Conv2d(ch[1], (self.no) * self.na, 1),
                                nn.Conv2d(ch[2], (self.no) * self.na, 1)])  # output conv

        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.id_embedding = id_embedding
        self.k = Parameter(torch.ones(1) * 10)

    def forward(self, x):

        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            # print(x[i][0].size())

            x[i] = self.m[i](x[i][0])  # conv
            # x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                # if self.inplace:
                #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    # y = torch.cat((xy, wh, y[..., 4:]), -1)
                y[..., 0:2] = ((y[..., 0:2] - 0.5) * self.k + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y = y[..., :6]
                z.append(y.view(bs, -1, self.no))

        # return x if self.training else (torch.cat(z, 1), x)
        return [x,self.k] if self.training else (torch.cat(z, 1), [x,self.k])


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
'''

class DenseMask(nn.Module):
    def __init__(self, mask=1, ch=()):
        super(DenseMask, self).__init__()
        self.proj1 = Conv(ch[0] // 2, 1, k=3)
        self.proj2 = nn.ConvTranspose2d(ch[1], 1, 4, stride=2,
                                        padding=1, output_padding=0,
                                        groups=1, bias=False)
        self.proj3 = nn.ConvTranspose2d(ch[2], 1, 8, stride=4,
                                        padding=2, output_padding=0,
                                        groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, layers):
        return self.sigmoid(self.proj1(layers[0][0]) + self.proj2(layers[1][0]) + self.proj3(layers[2][0]))


class SAAN(nn.Module):
    def __init__(self, id_embedding=256, ch=()):
        super(SAAN, self).__init__()
        self.proj1 = nn.Sequential(Conv(ch[0] // 2, 256, k=3),
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

        self.node = nn.Sequential(SAAN_Attention(k_size=3, ch=256 * 3, s_state=False, c_state=True),
                                  Conv(256 * 3, 256, k=3),
                                  nn.Conv2d(256, 512,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, layers):
        layers[0] = self.proj1(layers[0][1])
        layers[1] = self.proj2(layers[1][1])
        layers[2] = self.proj3(layers[2][1])
        # layers[0] = self.proj1(layers[0])
        # layers[1] = self.proj2(layers[1])
        # layers[2] = self.proj3(layers[2])
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
        # self.conv1 = Conv(ch, ch,k=1)

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
        b, c, h, w = x.size()

        # channel_attention
        if self.c_state:
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_c = self.c_attention(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) + \
                  self.c_attention(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y_c = self.sigmoid(y_c)

        # spatial_attention
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
        # self.independence = 0.7
        # self.share = 0.3
        self.w1 = Parameter(torch.ones(1) * 0.5)
        self.w2 = Parameter(torch.ones(1) * 0.5)
        w = 6
        h = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((w, h))

        self.c_attention1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.InstanceNorm2d(num_features=ch),
                                          nn.LeakyReLU(0.3, inplace=True))
        self.c_attention2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.InstanceNorm2d(num_features=ch),
                                          nn.LeakyReLU(0.3, inplace=True))

        self.sigmoid = nn.Sigmoid()
        # self.conv1 = Conv(ch, ch, k=1)
        # self.conv2 = Conv(ch, ch, k=1)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y_t1 = self.c_attention1(y)
        y_t2 = self.c_attention2(y)
        bs, c, h, w = y_t1.shape
        y_t1 = y_t1.view(bs, c, h * w)
        y_t2 = y_t2.view(bs, c, h * w)

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
        x_t1 = x_t1.contiguous().view(bs, c, h * w)
        x_t2 = x_t2.contiguous().view(bs, c, h * w)

        # x_t1 = torch.matmul(self.independence*M_t1 + self.share*M_s1, x_t1).contiguous().view(bs, c, h, w)
        # x_t2 = torch.matmul(self.independence*M_t2 + self.share*M_s2, x_t2).contiguous().view(bs, c, h, w)
        x_t1 = torch.matmul(self.w1 * M_t1 + (1 - self.w1) * M_s1, x_t1).contiguous().view(bs, c, h, w)
        x_t2 = torch.matmul(self.w2 * M_t2 + (1 - self.w2) * M_s2, x_t2).contiguous().view(bs, c, h, w)
        # print("M_t1",torch.sort(M_t1[0][0]))
        # print("y_t1",torch.max(y_t1),torch.min(y_t1))
        # print("y_t2", torch.max(y_t2), torch.min(y_t2))
        return [x_t1 + x, x_t2 + x]



class recheck_Box(nn.Module):
    def __init__(self,channal_base=256):
        super(recheck_Box, self).__init__()
        self.proj1 = Conv(channal_base, channal_base,k=3)
        self.proj2 = nn.ConvTranspose2d(channal_base*2, channal_base, 4, stride=2,
                                                       padding=1, output_padding=0,
                                                       groups=channal_base, bias=False)
        self.proj3 = nn.ConvTranspose2d(channal_base*4, channal_base, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=channal_base, bias=False)
        self.conv_box = nn.Sequential(Conv(channal_base, channal_base, k=3),
                                   nn.Conv2d(channal_base, 4, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self,layers):
        x_det = self.proj1(layers[0][0]) + self.proj2(layers[1][0]) + self.proj3(layers[2][0])
        x_box = self.conv_box(x_det)
        return x_box.permute(0,2,3,1)

class recheck_heatmap(nn.Module):
    def __init__(self,channal_base=256):
        super(recheck_heatmap, self).__init__()
        self.proj1 = Conv(channal_base, channal_base,k=3)
        self.proj2 = nn.ConvTranspose2d(channal_base*2, channal_base, 4, stride=2,
                                                       padding=1, output_padding=0,
                                                       groups=channal_base, bias=False)
        self.proj3 = nn.ConvTranspose2d(channal_base*4, channal_base, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=channal_base, bias=False)

        self.conv1 = nn.Sequential(Conv(1, channal_base,k=3),
                                  nn.Conv2d(channal_base, 1, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2 = nn.Sequential(Conv(channal_base, channal_base,k=3),
                                   Conv(channal_base, channal_base, k=3),
                                   nn.Conv2d(channal_base, 1, kernel_size=3, stride=1, padding=1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x_F,x_det):
        x_det = self.proj1(x_det[0][0]) + self.proj2(x_det[1][0]) + self.proj3(x_det[2][0])
        x_F = self.conv1(x_F)
        hmmap = self.sigmoid(self.conv2(x_F*x_det))
        return hmmap

class SiamMot(nn.Module):
    def __init__(self,opt,model):
        super(SiamMot, self).__init__()
        self.opt = opt
        channel_dict = {"s":128,"m":192,"l":256,"x":320}
        channal_base = channel_dict[opt.cfg.split(".")[-2][-1]]
        self.model = model
        self.stride = self.model.stride
        self.siambox = recheck_Box(channal_base=channal_base)
        self.siamheatmap = recheck_heatmap(channal_base=channal_base)
        self.r = self.opt.radius

    def forward(self,img,Tracklets_T=None,targets=None,previous_box=None):
        pred = self.model(img)
        if targets == None and Tracklets_T==None:
            return pred
        else:
            if targets != None:
                id_embeding, dense_mask, p, p_Det = pred[0], pred[1], pred[2][0], pred[2][3]
                batch_list = []
                id_F = []
                tcls, tbox, indices = build_targets_siammot(p, targets, self.model,train_state = True)
                b, gj, gi = indices[0]
                for b_i in range(len(p_Det)):
                    batch_list.append(sum(b == b_i))
                    p_Det_s = p_Det[b_i][p_Det[b_i][ :, 4] > self.opt.conf_thres]
                    dets, x_inds, y_inds = non_max_suppression_and_inds(p_Det_s.unsqueeze(0), self.opt.conf_thres, 0.5, dense_mask=dense_mask[b_i].unsqueeze(0),method='cluster_diou')
                    x_ori = gi[sum(batch_list[:len(batch_list) - 1]):sum(batch_list)]
                    y_ori = gj[sum(batch_list[:len(batch_list) - 1]):sum(batch_list)]
                    x_inds = torch.tensor(x_inds)
                    y_inds = torch.tensor(y_inds)
                    inds_add = torch.cat([x_inds.unsqueeze(0),y_inds.unsqueeze(0)],dim=0).permute(1,0).cuda()
                    inds = torch.cat([x_ori.unsqueeze(0),y_ori.unsqueeze(0)],dim=0).permute(1,0)
                    for i in range(len(inds_add)):
                        if inds_add[i] not in inds:
                            inds = torch.cat([inds,inds_add[i].unsqueeze(0)],dim=0)
                    inds = inds.long()
                    id_F.append(id_embeding[b_i][inds[:,1],inds[:,0]])

                '''
                tcls, tbox, indices = build_targets_siammot(p, targets, self.model,train_state = True)
                b, gj, gi = indices[0]
                id_f = id_embeding[indices[0]].detach()
                batch_list = []
                id_F = []
                for batch_i in range(max(b) + 1):
                    batch_list.append(sum(b == batch_i))
                    id_F.append(id_f[sum(batch_list[:len(batch_list) - 1]):sum(batch_list)])
                '''
                for batch_i in range(len(id_F) // 2):
                    id_F[batch_i * 2], id_F[batch_i * 2 + 1] = id_F[batch_i * 2 + 1], id_F[batch_i * 2]
                x_det = pred[2][2]

            if Tracklets_T != None:
                id_embeding = pred[0]
                _, train_out = pred[2]
                x_det = train_out[2]
                id_F = Tracklets_T

        for i in range(len(id_F)):
            h, w, c = id_embeding[i].size()
            k, c = id_F[i].size()
            y = torch.matmul(F.normalize(id_embeding[i].view(h * w, c), dim=1), F.normalize(id_F[i], dim=1).t()).view(h, w,k).permute(2, 0, 1)
            c, h, w = y.size()
            if Tracklets_T != None:
                y_mask = torch.zeros((c, h, w)).cuda()
            else:
                y_mask = torch.zeros((c,h,w)).cuda().half()
            y = torch.where(y > 0, y, y_mask)
            if self.opt.Global_Point:
                for j in range(len(y)):
                    y_ind = torch.nonzero(y[j] == torch.max(y[j]))[0]
                    y_mask[j][y_ind[0]-self.r:y_ind[0]+self.r, y_ind[1]-self.r:y_ind[1]+self.r] = 1
                y = y * y_mask

            y = torch.sum(y,dim=0)
            y = (y/torch.max(y)).unsqueeze(0).unsqueeze(0)
            if i == 0:
                x_F = y
            else:
                x_F = torch.cat([x_F, y], dim=0)
        if Tracklets_T == None and len(x_F) < len(id_embeding):
            x_F = torch.cat([x_F, torch.zeros((len(id_embeding)-len(x_F),1,h,w)).cuda()], dim=0)

        siambox = self.siambox(x_det)
        hmmap = self.siamheatmap(x_F,x_det)
        return [pred, hmmap, siambox]




class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # setting input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.out = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            # s = 256  # 2x min stride
            m.inplace = self.inplace

            x = self.forward(torch.zeros(2, ch, s, s))

            m.stride = torch.tensor([s / x.shape[-2] for x in x[2][0]])  # forward
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        # return self.forward_once(x, profile, visualize)  # single-scale inference, train
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        output = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            if m.i in self.out:
                output.append(x)
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return output

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)

    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    out_list = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args

        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:

            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            if i == 24:
                args = [c2, c1, *args[1:]]

            else:
                args = [c1, c2, *args[1:]]

            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            # c2 = sum([ch[x] for x in f])

        # elif m is Contract:
        #     c2 = ch[f] * args[0] ** 2
        # elif m is Expand:
        #     c2 = ch[f] // args[0] ** 2

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
        # if i == 0:
        #     ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), sorted(out_list)


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
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
