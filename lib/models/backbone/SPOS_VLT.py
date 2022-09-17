import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
from typing import Dict, List

from timm.models.layers import trunc_normal_

from torchvision.ops import roi_align
from transformers import BertTokenizer, BertModel

from models.sot.head import xcorr_depthwise
from models.sot.modules import NestedTensor
from models.sot.InMo.position_encoding import build_position_encoding

class roi_template(nn.Module):
    """
    template roi pooling: get 1*1 template
    """

    def __init__(self, roi_size=3, stride=8.0, inchannels=[64, 160, 320, 640], alpha=0.1):
        """
        Args:
            roi_size: output size of roi
            stride: network stride
            inchannels: input channels
            alpha: for leaky-relu
        """
        super(roi_template, self).__init__()
        self.roi_size, self.stride = roi_size, float(stride)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=roi_size, stride=1),
            nn.BatchNorm2d(inchannels),
            nn.LeakyReLU(alpha),
        )


        self.box_indices = None

    def forward(self, boxes, fea, feas3):
        """
        Args:
            boxes: [b, 4]
            fea: [b, c, h, w]
            target_fea: [b, c]

        Returns: cls results
        """

        fea = self.fea_encoder(fea)
        feas3 = self.fea_encoder_s3(feas3)

        B, _ = boxes.size()

        if self.box_indices is None:
            box_indices = torch.arange(B, dtype=torch.float32).reshape(-1, 1)
            self.box_indices = torch.tensor(box_indices, dtype=torch.float32)

        batch_index = self.box_indices.to(boxes.device)  # [K, 1]
        batch_box = torch.cat((batch_index, boxes), dim=1)
        # ROI pooling layer
        # print(fea.dtype, batch_box.dtype)
        pool_fea = roi_align(fea, batch_box, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                             sampling_ratio=-1)  # [K, C 3, 3]
        pool_fea_s3 = roi_align(feas3, batch_box, [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                sampling_ratio=-1)  # [K, C 3, 3]
        # spatial resolution to 1*1
        pool_fea = self.spatial_conv(pool_fea)  # [K, C]
        pool_fea_s3 = self.spatial_conv_s3(pool_fea_s3)  # [K, C]

        if len(pool_fea.size()) == 1:
            pool_fea = pool_fea.unsqueeze(0)
            pool_fea_s3 = pool_fea_s3.unsqueeze(0)

        return pool_fea, pool_fea_s3


class Shufflenet(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride, mapping):
        super(Shufflenet, self).__init__()
        self.stride = stride
        self.mapping = mapping
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.base_mid_channel = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize,
                      stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2 or mapping:
            branch_proj = [
                # dw
                nn.Conv2d(
                    inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 2 or self.mapping:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        elif self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, stride, mapping):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]

        self.base_mid_channel = mid_channels
        self.stride = stride
        self.mapping = mapping
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp, affine=False),
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
        ]

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2 or self.mapping:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 2 or self.mapping:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        elif self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]



class SPOS_VLT(nn.Module):

    def __init__(self, input_size=224, n_class=1000, nlp=True):
        super(SPOS_VLT, self).__init__()

        assert input_size % 32 == 0
        self.nlp = nlp

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            nn.ReLU(inplace=True),
        )

        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                # if i == 0:
                #     inp, outp, stride = input_channel, output_channel, 2
                # else:
                #     inp, outp, stride = input_channel // 2, output_channel, 1
                mapping = False
                if i == 0:
                    if idxstage < 2:
                        inp, outp, stride = input_channel, output_channel, 2
                    else:
                        mapping = True
                        inp, outp, stride = input_channel, output_channel, 1
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels)
                archIndex += 1
                self.features.append(torch.nn.ModuleList())
                for blockIndex in range(4):
                    if blockIndex == 0:
                        # print('Shuffle3x3')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride, mapping=mapping))
                    elif blockIndex == 1:
                        # print('Shuffle5x5')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride, mapping=mapping))
                    elif blockIndex == 2:
                        # print('Shuffle7x7')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride, mapping=mapping))
                    elif blockIndex == 3:
                        # print('Xception')
                        self.features[-1].append(
                            Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride, mapping=mapping))
                    else:
                        raise NotImplementedError
                input_channel = output_channel

        self.archLen = archIndex
        # self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))

        if nlp:
            count = 0
            self.nlp_layers = nn.ModuleList()
            for i in self.stage_repeats:
                count += i
                self.nlp_layers.append(copy.deepcopy(self.features[count-1]))

        alpha = 0.1
        self.vistoken_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(alpha),
            ),
            nn.Sequential(
                nn.Conv2d(160, 160, kernel_size=3, stride=1),
                nn.BatchNorm2d(160),
                nn.LeakyReLU(alpha),
            ),
            nn.Sequential(
                nn.Conv2d(320, 320, kernel_size=3, stride=1),
                nn.BatchNorm2d(320),
                nn.LeakyReLU(alpha),
            ),
            nn.Sequential(
                nn.Conv2d(640, 640, kernel_size=3, stride=1),
                nn.BatchNorm2d(640),
                nn.LeakyReLU(alpha),
            ),
        ])
        self.roi_size = 3
        self.stride = 8

        self._initialize_weights()

    def forward_track(self, x, architecture, out_dices=[1,2,3], out_mark=[4,8,16,20]):
        assert self.archLen == len(architecture)
        out = []

        x = self.first_conv(x)
        # print('first',x.shape)
        count = 0
        for archs, arch_id in zip(self.features, architecture):
            # if arch_id != 4:

            x = archs[arch_id](x)
            count += 1
            if count in out_mark:
                out.append(x)

        out = [out[i] for i in out_dices]
        return out

    def forward_track_nlp_tem(self, x, architecture, nlp_tokens=None, nlp_cand=None, out_dices=[1,2,3], out_mark=[4,8,16,20], out_mark_nlp=[3,7,15,19], batch_box=None):
        assert self.archLen == len(architecture)
        if nlp_cand is None:
            nlp_cand = [np.random.randint(4) for i in range(4)]
        out = []
        out_cv = []
        out_tem = []

        x = self.first_conv(x)
        count = 0
        batch_box_5 = None
        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x)
            count += 1
            if count in out_mark:
                x = x + out_cv[-1]
                out.append(x)
            if count in out_mark_nlp:
                index = out_mark_nlp.index(count)

                B = x.size(0)

                if batch_box_5 is None:
                    box_indices = torch.arange(B, dtype=torch.float32).reshape(-1, 1).to(x.device)
                    batch_box_5 = torch.cat((box_indices, batch_box), dim=1)
                # ROI pooling layer
                pool_fea = roi_align(x, batch_box_5.float(), [self.roi_size, self.roi_size], spatial_scale=1. / self.stride,
                                     sampling_ratio=-1)  # [K, C 3, 3]
                # spatial resolution to 1*1
                pool_fea = self.vistoken_layers[index](pool_fea)  # [K, C]
                if len(pool_fea.size()) == 1:
                    pool_fea = pool_fea.unsqueeze(0)
                out_tem.append(pool_fea)

                x_cv = xcorr_depthwise(x, pool_fea)
                out_cv.append(self.nlp_layers[index][nlp_cand[index]](x_cv))
        self.out_tem = out_tem

        out = [out[i] for i in out_dices]
        return out

    def forward_track_nlp_sear(self, x, architecture, nlp_tokens=None, nlp_cand=None, out_dices=[1,2,3], out_mark=[4,8,16,20], out_mark_nlp=[3,7,15,19], batch_box=None):
        assert self.archLen == len(architecture)
        if nlp_cand is None:
            nlp_cand = [np.random.randint(4) for i in range(4)]
        out = []
        out_cv = []

        x = self.first_conv(x)
        count = 0
        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x)
            count += 1
            if count in out_mark:
                x = x + out_cv[-1]
                out.append(x)
            if count in out_mark_nlp:
                index = out_mark_nlp.index(count)

                x_cv = xcorr_depthwise(x, self.out_tem[index])
                out_cv.append(self.nlp_layers[index][nlp_cand[index]](x_cv))

        out = [out[i] for i in out_dices]
        return out

    def forward_track_nlp(self, x, architecture, nlp_tokens=None, nlp_cand=None, out_dices=[1,2,3], out_mark=[4,8,16,20], out_mark_nlp=[3,7,15,19]):
        assert self.archLen == len(architecture)
        if nlp_cand is None:
            nlp_cand = [np.random.randint(4) for i in range(4)]
        out = []
        out_nlp = []

        x = self.first_conv(x)
        count = 0
        for archs, arch_id in zip(self.features, architecture):
            # if arch_id != 4:
            x = archs[arch_id](x)
            count += 1
            if count in out_mark:
                x = x + out_nlp[-1]
                out.append(x)
            if count in out_mark_nlp:
                index = out_mark_nlp.index(count)
                x_nlp = xcorr_depthwise(x, nlp_tokens[index])
                out_nlp.append(self.nlp_layers[index][nlp_cand[index]](x_nlp))

        out = [out[i] for i in out_dices]
        return out

    def forward(self, x, architecture):
        assert self.archLen == len(architecture)

        x = self.first_conv(x)

        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x)

        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def nas(self, nas_ckpt_path=None):
        supernet_state_dict = torch.load(nas_ckpt_path)['state_dict']
        new_dict = {}
        for key, value in supernet_state_dict.items():
            new_dict[key[7:]] = value
        self.load_state_dict(new_dict, strict=False)

    def load_nlp(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")  # .cuda()
        self.token_map = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Conv2d(768, 160, kernel_size=1, bias=False),
                nn.BatchNorm2d(160),
            ),
            nn.Sequential(
                nn.Conv2d(768, 320, kernel_size=1, bias=False),
                nn.BatchNorm2d(320),
            ),
            nn.Sequential(
                nn.Conv2d(768, 640, kernel_size=1, bias=False),
                nn.BatchNorm2d(640),
            )
        ])
        self.token_map.apply(self._init_weights)

    def forward_nlp(self, phrase):
        # phrase = "I'm not sure, this can work, lol -.-"
        # phrase = "I'm not sure"
        # phrase = [phrase]
        # tokens = self.bert_tokenizer.batch_encode_plus(phrase, padding='longest', return_tensors='pt')
        embeds = self.bert_model(phrase[0], attention_mask=phrase[1])[0]
        embeds = torch.mean(embeds, dim=1).reshape([-1, 768, 1, 1])
        # embeds = self.down_nlp(embeds)
        embeds = [layer(embeds) for layer in self.token_map]
        return embeds


class Backbone_SPOS_VLT(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, config):
        super().__init__()

        self.body = SPOS_VLT()
        self.num_channels = config.BACKBONE.CHANNEL

    def forward(self, tensor_list, nas_lists=None, nlp_tokens=None, vistoken=False, batch_box=None, nlp_cand=None):
        input_img = tensor_list

        if vistoken:
            if input_img.shape[-1] > 200:
                # xs = self.body(input_img, nas_lists[1])
                xs = self.body.forward_track_nlp_sear(input_img, nas_lists[1], None, nlp_cand=nlp_cand)
            else:
                # xs = self.body.forward_track(input_img, nas_lists[0])
                xs = self.body.forward_track_nlp_tem(input_img, nas_lists[0], None, batch_box=batch_box, nlp_cand=nlp_cand)
        else:
            if input_img.shape[-1] > 200:
                # xs = self.body(input_img, nas_lists[1])
                xs = self.body.forward_track_nlp(input_img, nas_lists[1], nlp_tokens, nlp_cand=nlp_cand)
            else:
                # xs = self.body.forward_track(input_img, nas_lists[0])
                xs = self.body.forward_track_nlp(input_img, nas_lists[0], nlp_tokens, nlp_cand=nlp_cand)
        return xs


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.pos_z = None
        self.pos_x = None

    def nas(self, nas_ckpt_path=None):
        self[0].body.nas(nas_ckpt_path=nas_ckpt_path)

    def load_nlp(self):
        self[0].body.load_nlp()

    def forward(self, tensor_list, nas_lists=None, nlp_tokens=None, vistoken=False, batch_box=None, nlp_cand=None):
        xs = self[0](tensor_list, nas_lists=nas_lists, nlp_tokens=nlp_tokens, vistoken=vistoken, batch_box=batch_box, nlp_cand=nlp_cand)[
            -1]
        if tensor_list.size(-1) > 200:
            # xs = xs[:,:,1:31,1:31]
            if self.pos_x is None or self.pos_x.size(0) != xs.size(0):
                self.pos_x = self[1](xs)
            pos = self.pos_x.to(xs.device)
        else:
            # xs = xs[:, :, 1:15, 1:15]
            # xs = xs[:,:,4:12,4:12]
            if self.pos_z is None or self.pos_z.size(0) != xs.size(0):
                self.pos_z = self[1](xs)
            pos = self.pos_z.to(xs.device)
        return xs, [pos]


def SPOS_VLT_Wrapper(config):
    position_embedding = build_position_encoding(config.MODEL)
    backbone = Backbone_SPOS_VLT(config.MODEL)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == "__main__":
    cand = ((2, 1, 0, 2, 3, 2, 1, 1, 1, 1, 0, 1, 3, 2, 1, 2, 3, 2, 1, 2),
            (0, 3, 3, 0, 1, 1, 0, 3, 1, 0, 0, 1, 2, 3, 3, 3, 1, 3, 0, 0))
    cand = [[0 for i in range(20)], [0 for i in range(20)]]
    architecture = cand[1]
    # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    # scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    channels_scales = []
    # for i in range(len(scale_ids)):
    #     channels_scales.append(scale_list[scale_ids[i]])
    model = SPOS_VLT()
    # print(model)
    # for name, m in model.named_children():
    #     print(name)
    state = model.state_dict()
    keys = state.keys()
    nlp_layers = []
    features = []
    for key in keys:
        if 'nlp_layers' in key:
            nlp_layers.append(key)
        elif 'features' in key:
            features.append(key)
    new_dict = {}
    for key, value in state.items():
        if 'nlp_layers' in key:
            continue
        elif 'features.3' in key:
            new_dict[key] = value
            new_dict['nlp_layers.0'+key[10:]] = value
            # print(key, state['nlp_layers.0'+key[10:]].shape, value.shape)
        elif 'features.7' in key:
            new_dict[key] = value
            new_dict['nlp_layers.1'+key[10:]] = value
            # print(key, state['nlp_layers.1' + key[10:]].shape, value.shape)
        elif 'features.15' in key:
            new_dict[key] = value
            new_dict['nlp_layers.2'+key[11:]] = value
        elif 'features.19' in key:
            new_dict[key] = value
            new_dict['nlp_layers.3'+key[11:]] = value
        else:
            new_dict[key] = value

    print(nlp_layers)
    print(features)
    model.load_state_dict(new_dict, strict=True)

    test_data = torch.rand(2, 3, 255, 255)
    test_outputs = model.forward_track(test_data, architecture)
    for i in test_outputs:
        print(i.shape)

    test_data = torch.rand(2, 3, 127, 127)
    architecture = cand[0]
    test_outputs = model.forward_track(test_data, architecture)
    for i in test_outputs:
        print(i.shape)
    # print(test_outputs.size())
