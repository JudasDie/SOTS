# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional
import matplotlib.pyplot as plt
import cv2
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block as TimmBlock


class WindowAttentionCross(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, dropout=0., proj_drop=0., ws=8):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q_map = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_map = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.group = ws
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B, len_q, len_kv, C = q.size(0), q.size(1), kv.size(1), q.size(2)
        if math.sqrt(len_q) < self.group or math.sqrt(len_kv) < self.group:
            group = 0
        else:
            group = self.group
        q = self.q_map(q)
        kv = self.kv_map(kv).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        if group > 0:
            # B N(HW) C

            hw_q = math.sqrt(len_q)
            hw_k = math.sqrt(len_kv)
            hw_v = math.sqrt(len_kv)
            q_group, k_group, v_group = int(hw_q // group), int(hw_k // group), int(hw_v // group)

            group_square = group * group

            q = q.reshape(B, group, q_group, group, q_group, C).transpose(2, 3).reshape(B, group_square,
                                                                                        q_group * q_group,
                                                                                        self.num_heads,
                                                                                        self.head_dim).transpose(2, 3)
            k = k.reshape(B, group, k_group, group, k_group, C).transpose(2, 3).reshape(B, group_square,
                                                                                        k_group * k_group,
                                                                                        self.num_heads,
                                                                                        self.head_dim).transpose(2, 3)
            v = v.reshape(B, group, v_group, group, v_group, C).transpose(2, 3).reshape(B, group_square,
                                                                                        v_group * v_group,
                                                                                        self.num_heads,
                                                                                        self.head_dim).transpose(2, 3)
        else:
            # B N(HW) C
            B, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

            q = q.view(B, len_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, len_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, len_v, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        if group > 0:
            q = (attn @ v).transpose(1, 2).reshape(B, group, group, q_group, q_group, C).transpose(2, 3).reshape(B, len_q, -1)
        else:
            q = (attn @ v).transpose(1, 2).reshape(B, len_q, -1)

        q = self.fc(q)
        q = self.proj_drop(q)
        return q


class WindowBlockCross(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0., proj_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super().__init__()

        self.attn = WindowAttentionCross(dim, num_heads, qkv_bias, qk_scale, dropout, proj_drop, ws)
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)

    def forward(self, q, kv):
        x = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, dropout=0., proj_drop=0., ws=8):
        assert ws != 1
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class WindowBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0., proj_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super().__init__()
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, dropout, proj_drop, sr_ratio)
        else:
            self.attn = WindowAttention(dim, num_heads, qkv_bias, qk_scale, dropout, proj_drop, ws)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=256, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu", pos_mode='cnn', ws=16, backbone_channel=256):
        super().__init__()

        self.encoder = Encoder(d_model, nhead, dim_feedforward, dropout, activation, num_featurefusion_layers, pos_mode, ws)
        self.decoder = Decoder(d_model, nhead, dim_feedforward, dropout, activation, pos_mode, ws)

        self.input_proj = nn.Conv2d(backbone_channel, d_model, kernel_size=1)

        # self._reset_parameters()
        self.d_model = d_model
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

    def forward(self, src_temp, mask_temp=None, src_search=None, mask_search=None, pos_temp=None, pos_search=None):
        src_temp = self.input_proj(src_temp)
        src_search = self.input_proj(src_search)

        src_temp = src_temp.flatten(2).transpose(1, 2)
        src_search = src_search.flatten(2).transpose(1, 2)
        # mask_temp = mask_temp.flatten(1)
        # mask_search = mask_search.flatten(1)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search)
        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=mask_search,
                          memory_key_padding_mask=mask_temp)
        return hs.unsqueeze(0)


class Decoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, pos_mode='cnn', ws=8):
        super().__init__()

        self.layers = nn.ModuleList([DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                        ws=1 if i % 2 == 1 else ws) for i in range(1)])

        self.pos_mode = pos_mode

        if pos_mode == 'embed':
            self.pos_embed_z = nn.Parameter(torch.zeros(1, 256, d_model))
            self.pos_embed_x = nn.Parameter(torch.zeros(1, 1024, d_model))
            self.pos_drop_z = nn.Dropout(p=dropout)
            self.pos_drop_x = nn.Dropout(p=dropout)
            trunc_normal_(self.pos_embed_z, std=.02)
            trunc_normal_(self.pos_embed_x, std=.02)
        elif pos_mode == 'cnn':
            self.pos_block_z = PosCNN(d_model, d_model)
            self.pos_block_x = PosCNN(d_model, d_model)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output2 = tgt
        output1 = memory

        for layer in self.layers:
            if self.pos_mode == 'embed':
                output1 = output1 + self.pos_embed_z
                output2 = output2 + self.pos_embed_x
                output1 = self.pos_drop_z(output1)
                output2 = self.pos_drop_x(output2)
            elif self.pos_mode == 'cnn':
                output1 = self.pos_block_z(output1)
                output2 = self.pos_block_x(output2)
            output = layer(output2, output1, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        return output


class Encoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, pos_mode='cnn', ws=8):
        super().__init__()

        self.layers = nn.ModuleList([FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                        # ws) for i in range(num_layers)])
                                                        ws=1 if i % 2 == 1 else ws) for i in range(num_layers)])  # for OTB 70.9

        self.num_layers = num_layers
        self.pos_mode = pos_mode

        if pos_mode == 'embed':
            self.pos_embed_z = nn.Parameter(torch.zeros(1, 256, d_model))
            self.pos_embed_x = nn.Parameter(torch.zeros(1, 1024, d_model))
            self.pos_drop_z = nn.Dropout(p=dropout)
            self.pos_drop_x = nn.Dropout(p=dropout)
            trunc_normal_(self.pos_embed_z, std=.02)
            trunc_normal_(self.pos_embed_x, std=.02)
        elif pos_mode == 'cnn':
            self.pos_block_z = PosCNN(d_model, d_model)
            self.pos_block_x = PosCNN(d_model, d_model)

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2

        for idx, layer in enumerate(self.layers):
            if self.pos_mode == 'embed':
                output1 = output1 + self.pos_embed_z
                output2 = output2 + self.pos_embed_x
                output1 = self.pos_drop_z(output1)
                output2 = self.pos_drop_x(output2)
            elif self.pos_mode == 'cnn':
                output1 = self.pos_block_z(output1)
                output2 = self.pos_block_x(output2)
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", ws=8):
        super().__init__()

        self.cross_attn = WindowAttentionCross(d_model, nhead, dropout=dropout, ws=ws)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)


    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None, visualize=True):

        tgt2 = self.cross_attn(tgt, memory)
        tgt = self.norm1(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", ws=8):
        super().__init__()
        self.self_attn1 = WindowBlock(d_model, nhead, dropout=dropout, ws=ws)
        self.self_attn2 = WindowBlock(d_model, nhead, dropout=dropout, ws=ws)
        self.cross_attn1 = WindowBlockCross(d_model, nhead, dropout=dropout, ws=ws)
        self.cross_attn2 = WindowBlockCross(d_model, nhead, dropout=dropout, ws=ws)

        self.linear1_z = nn.Linear(d_model, d_model)
        self.linear1_x = nn.Linear(d_model, d_model)
        self.norm1_z = nn.LayerNorm(d_model)
        self.norm1_x = nn.LayerNorm(d_model)
        self.dropout1_z = nn.Dropout(dropout)
        self.dropout1_x = nn.Dropout(dropout)

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None, visualize=True):

        src1 = self.self_attn1(src1)
        src2 = self.self_attn2(src2)

        # if visualize:
        #     vis_list = []  # vis
        #     vis_list.append(src12)  # vis
        #     vis_list.append(src22)  # vis

        src12 = self.cross_attn1(src1, src2)
        src22 = self.cross_attn2(src2, src1)
        # if visualize:
        #     vis_list.append(src12)  # vis
        #     vis_list.append(src22)  # vis

        src1 = self.linear1_z(src1 + self.dropout1_z(src12))
        src1 = self.norm1_z(src1)

        src2 = self.linear1_x(src2 + self.dropout1_x(src22))
        src2 = self.norm1_x(src2)

        # if visualize:
        #     len_template = int(math.sqrt(src1.size(1)))
        #     len_search = int(math.sqrt(src2.size(1)))
        #     plt.figure()
        #     plt.subplot(221)
        #     plt.imshow(cv2.resize(
        #         vis_list[0].detach().mean(dim=-1).squeeze().reshape(len_template, len_template).cpu().numpy(),
        #         (256, 256)))
        #     plt.subplot(222)
        #     plt.imshow(cv2.resize(
        #         vis_list[1].detach().mean(dim=-1).squeeze().reshape(len_search, len_search).cpu().numpy(),
        #         (256, 256)))
        #     plt.subplot(223)
        #     plt.imshow(cv2.resize(
        #         vis_list[2].detach().mean(dim=-1).squeeze().reshape(len_template, len_template).cpu().numpy(),
        #         (256, 256)))
        #     plt.subplot(224)
        #     plt.imshow(cv2.resize(
        #         vis_list[3].detach().mean(dim=-1).squeeze().reshape(len_search, len_search).cpu().numpy(),
        #         (256, 256)))
        #     plt.show()
        #     plt.close()

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def Trans_Fuse(config):
    return FeatureFusionNetwork(
        d_model=config.NECK.HIDDEN_DIM,
        dropout=config.NECK.DROPOUT,
        nhead=config.NECK.NHEAD,
        dim_feedforward=config.NECK.DIM_FEEDFORWARD,
        num_featurefusion_layers=config.NECK.LAYER_NUM,
        activation=config.NECK.ACTIVATION,
        ws=config.NECK.WINDOW_SIZE,
        backbone_channel=config.BACKBONE.CHANNEL
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    module = FeatureFusionNetwork(
        d_model=256,
        dropout=0.,
        nhead=4,
        dim_feedforward=1024,
        num_featurefusion_layers=4
    )
    input_z = torch.rand([2, 256, 16, 16])
    input_x = torch.rand([2, 256, 32, 32])
    output = module(input_z, None, input_x)
    print(output.shape)
