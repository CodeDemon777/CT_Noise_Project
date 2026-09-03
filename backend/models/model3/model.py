import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=3, s=stride, p=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ASPP(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.b1 = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.b2 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=rates[0], d=rates[0])
        self.b3 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=rates[1], d=rates[1])
        self.b4 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=rates[2], d=rates[2])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        f4 = self.b4(x)
        fg = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([f1, f2, f3, f4, fg], dim=1))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(1, 32, 3, 1, 1),
            ConvBNReLU(32, 64, 3, 1, 1)
        )
        down2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=down2),
            BasicBlock(128, 128, stride=1)
        )
        down3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=down3),
            BasicBlock(256, 256, stride=1)
        )
        down4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.stage4 = nn.Sequential(
            BasicBlock(256, 256, stride=2, downsample=down4),
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 256, stride=1)
        )
        down5 = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.stage5 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=down5),
            BasicBlock(512, 512, stride=1),
            BasicBlock(512, 512, stride=1)
        )
        self.proj_high = ConvBNReLU(512, 256, 1, 1, 0)
        self.proj_low = ConvBNReLU(128, 64, 1, 1, 0)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        low = self.proj_low(x2)
        high = self.proj_high(x5)
        return low, high


class Decoder(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decode = nn.Sequential(
            ConvBNReLU(304, 256, 3, 1, 1),
            nn.Dropout(0.5),
            ConvBNReLU(256, 256, 3, 1, 1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, low, high, orig_size):
        low_feat = self.low_proj(low)
        high_up = F.interpolate(high, size=low_feat.shape[2:], mode="bilinear", align_corners=False)
        concat = torch.cat([high_up, low_feat], dim=1)
        out = self.decode(concat)
        out = F.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)
        return out


class DeepLabV3Plus(nn.Module):
    """
    Model 3: DeepLabV3+ with custom ResNet encoder and ASPP.
    Detects Salt & Pepper and Random-Valued Impulse Noise (RVIN).
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = Encoder()
        self.aspp = ASPP(256, 256)
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        orig_size = x.shape[2:]
        low, high = self.encoder(x)
        aspp_out = self.aspp(high)
        return self.decoder(low, aspp_out, orig_size)
