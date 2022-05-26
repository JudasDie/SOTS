import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import matplotlib.pyplot as plt
import cv2


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, mode='transformer', cross=False):
        super().__init__()

        if mode == 'performer':
            self.attn = Block_performer(d_model, n_head)
        else:
            self.attn = Block_transformer_cat(d_model, n_head, cross=cross)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.attn(query, key, value, key_padding_mask)


class Block_transformer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5

        self.fc = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # B N(HW) C
        B, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = q.view(B, len_q, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, len_k, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, len_v, self.n_head, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)  # -1e9 / 0

        attn = F.softmax(attn, dim=-1)
        q = (attn @ v).transpose(1, 2).reshape(B, len_q, -1)
        q = self.fc(q)
        return q


class Block_transformer_cat(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, cross=False, group=-1):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.cross = cross
        if cross:
            self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.group = group


    def forward(self, q, k, v, mask=None, visualize=False):
        if self.cross:
            B, len_q = q.size(0), q.size(1)
            N = len_q + k.size(1) + v.size(1)
            x = torch.cat([q,k,v], dim=1)
            qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4).transpose(2, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
        elif self.group > 0:
            # B N(HW) C
            B, len_q, len_k, len_v, C = q.size(0), q.size(1), k.size(1), v.size(1), q.size(2)
            hw_q = math.sqrt(len_q)
            hw_k = math.sqrt(len_k)
            hw_v = math.sqrt(len_v)
            q_group, k_group, v_group = hw_q // self.group, hw_k // self.group, hw_v // self.group

            group_square = self.group * self.group

            q = q.reshape(B, q_group, self.group, q_group, self.group, C).transpose(2, 3).reshape(B, q_group*q_group, group_square, self.n_head, self.d_head).transpose(2, 3)
            k = k.reshape(B, k_group, self.group, k_group, self.group, C).transpose(2, 3).reshape(B, k_group*k_group, group_square, self.n_head, self.d_head).transpose(2, 3)
            v = v.reshape(B, v_group, self.group, v_group, self.group, C).transpose(2, 3).reshape(B, v_group*v_group, group_square, self.n_head, self.d_head).transpose(2, 3)
        else:
            # B N(HW) C
            B, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

            q = q.view(B, len_q, self.n_head, self.d_head).transpose(1, 2)
            k = k.view(B, len_k, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, len_v, self.n_head, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None and self.group < 0:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, -1e9)  # -1e9 / 0

        attn = F.softmax(attn, dim=-1)

        if self.cross:
            q = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            q = q[:, :len_q]
        elif self.group > 0:
            q = (attn @ v).attn.transpose(2, 3).reshape(B, q_group, q_group, self.group, self.group, C).transpose(2, 3).reshape(B, len_q, -1)
        else:
            q = (attn @ v).transpose(1, 2).reshape(B, len_q, -1)

        q = self.fc(q)
        # if visualize:
        #     plt.figure()
        #     plt.imshow(cv2.resize(q.detach().mean(dim=-1).squeeze().reshape(int(math.sqrt(len_q)), int(math.sqrt(len_q))).cpu().numpy(), (256,256)))
        #     plt.show()
        #     plt.close()
        return q


class Block_performer(nn.Module):
    def __init__(self, d_model, n_head=1, kernel_ratio=0.5):
        super().__init__()
        self.emb = d_model * 1  # we use 1, so it is no need here
        self.proj = nn.Linear(self.emb, self.emb, bias=False)
        self.n_head = n_head
        self.epsilon = 1e-8  # for stable in division

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, q, k, v, mask=None):
        # k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)

        # if mask is not None and mask.size(-1) == D.size(-1):
        #     D = D.masked_fill(mask.unsqueeze(-1), -1e9)  # -1e9 / 0
        #     D = F.softmax(D, dim=-1)

        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (
                    D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = self.proj(y)  # not same as token_transformer in T2T layer, use q as skip connection

        return y



