''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: matching operators for Siamese tracking
Data: 2021.6.23
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

alpha = 0.1
class PointDW(nn.Module):
    """
    point (1*1) depthwise correlation used in AutoMatch: keep resolution after matching
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, inchannels=256, outchannels=256):
        super(PointDW, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature

    def forward(self, xf, zf):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]

        xf2 = self.s_embed(xf)
        zf2 = self.t_embed(zf)

        merge = xf2 * zf2
        return merge


class Concat(nn.Module):
    """
    Concatenation used in AutoMatch
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, inchannels=256, outchannels=256):
        super(Concat, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature
        self.down = nn.Conv2d(outchannels * 2, outchannels, 1)  # embeeding for template feature

    def forward(self, xf, zf):

        xf2 = self.s_embed(xf)
        B, C, H, W = xf2.size()

        zf2 = self.t_embed(zf)
        zf2 = zf2.repeat(1, 1, H, W)

        merge = torch.cat((xf2, zf2), dim=1)
        merge = self.down(merge)

        return merge


class PointAdd(nn.Module):
    """
    pointwise (1*1) addition used in AutoMatch
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, inchannels=256, outchannels=256):
        super(PointAdd, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature

    def forward(self, xf, zf):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]

        xf2 = self.s_embed(xf)
        zf2 = self.t_embed(zf)

        merge = xf2 + zf2
        return merge


class PairRelation(nn.Module):
    """
    Pairwise Relation used in AutoMatch
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, inchannels=256, outchannels=256):
        super(PairRelation, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature
        self.down = nn.Conv2d(15*15, 256, 1)  # embeeding for template feature


    def forward(self, xf, zf):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]

        xf2 = self.s_embed(xf)  # [B, C, H, W]
        zf2 = self.s_embed(zf)

        B, C, H, W = xf2.size()
        xf2 = xf2.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        xf2 = xf2.view(B, -1, C)  # [B, H*W, C]
        zf2 = zf2.view(B, C, -1)  # [B, C, H*W]

        pc = torch.matmul(xf2, zf2)  # [HW, HW]
        pc = pc.permute(0, 2, 1).view(B, -1, H, W)
        pc = self.down(pc)

        return pc


class FiLM(nn.Module):
    """
    FiLM in AutoMatch
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """
    def __init__(self, inchannels=256, outchannels=256):
        super(FiLM, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.conv_g = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=1),
            nn.LeakyReLU(alpha))

        self.conv_b = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=1),
            nn.LeakyReLU(alpha))

    def generate_mult_bias(self, fea):
        # for s4
        fm = self.conv_g(fea)
        fb = self.conv_b(fea)
        return fm, fb

    def infuse(self, fea, target):
        mult, bias = target

        yfa = (1 + mult) * fea + bias

        return yfa

    def forward(self, xf, zf):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]

        xf = self.s_embed(xf)
        fm, fb = self.generate_mult_bias(zf)
        merge = self.infuse(xf, [fm, fb])

        return merge


class SimpleSelfAtt(nn.Module):
    """
    self-attention encoder used in AutoMatch
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """

    def __init__(self, inchannels=256, outchannels=256):
        super(SimpleSelfAtt, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed_v = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature

        self.trans = nn.MultiheadAttention(outchannels, 4)

    def forward(self, xf, zf):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]
        xf2 = self.s_embed(xf)
        zf_value = self.t_embed_v(zf)
        zf2 = self.t_embed(zf)
        B, C, Hx, Wx = xf2.size()
        _, _, Hz, Wz = zf2.size()

        xf2 = xf2.view(B, C, -1).permute(2, 0, 1).contiguous()
        zf2 = zf2.view(B, C, -1).permute(2, 0, 1).contiguous()
        v = zf_value.view(B, C, -1).permute(2, 0, 1).contiguous()

        merge, weights = self.trans(xf2, zf2, v)
        merge = merge.permute(1, 2, 0).contiguous().view(B, -1, Hx, Wx)
        return merge

class TransGuide(nn.Module):
    """
    Transductive Guidance in AutoMatch: (coarse) mask label propagation
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """
    def __init__(self, inchannels=256, outchannels=256):
        super(TransGuide, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature

    def forward(self, xf, zf, zf_mask):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]

        xf2 = self.s_embed(xf)
        zf2 = self.t_embed(zf)

        B, C, Hx, Wx = xf2.size()
        B, C, Hz, Wz = zf2.size()

        xf2 = xf2.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        xf2 = xf2.view(B, -1, C)   # [B, H*W, C]
        zf2 = zf2.view(B, C, -1)   # [B, C, H*W]

        att = torch.matmul(xf2, zf2)  # [HW, HW]
        att = att / math.sqrt(C)
        att = F.softmax(att, dim=-1)  # [HW, HW]
        zf_mask = nn.Upsample(size=(Hz, Wz), mode='bilinear', align_corners=True)(zf_mask.unsqueeze(1))
        # zf_mask = (zf_mask > 0.5).float()
        zf_mask = zf_mask.view(B, -1, 1)

        arn = torch.matmul(att, zf_mask)  # [B, H*W]
        arn = arn.view(B, Hx, Hx).unsqueeze(1)

        merge = xf + arn
        return merge



