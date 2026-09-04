"""RRDBNet A2V5B — 기업 통합 노트북의 인라인 정의를 그대로 이식.

실사 근거: 내부 `통합 파이프라인.ipynb` cell 16
(`%%writefile .../rrdb_model_vr2.py`). 레포 루트의 `rrdb_model.py`(48ch·23블록·x4)와
아키텍처가 다르며, 최종 가중치 `A2_v5b_rrdbnet_best.pth`에 맞는 것은 이 8블록·64ch·
no-upscale 정의다(기업 문의 11번 항목).
"""
import torch
import torch.nn as nn


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class RRDBNetA2V5B(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB() for _ in range(8)])
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk
        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return out.clamp(0, 1)
