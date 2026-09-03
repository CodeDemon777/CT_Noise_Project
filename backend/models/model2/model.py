import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(F_int)
        )

        self.W_l = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g_in = self.W_g(g)
        x_in = self.W_l(x)

        if g_in.shape[2:] != x_in.shape[2:]:
            g_in = TF.resize(
                g_in,
                size=x_in.shape[2:]
            )

        out = self.relu(g_in + x_in)

        attention_map = self.psi(out)

        return x * attention_map


class AttentionUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        features=[64, 128, 256, 512]
    ):
        super(AttentionUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        for feature in features:

            self.downs.append(
                DoubleConv(
                    in_channels,
                    feature
                )
            )

            in_channels = feature

        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2
        )

        self.att_gates = nn.ModuleList()

        for feature in reversed(features):

            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )

            self.att_gates.append(
                AttentionGate(
                    F_g=feature,
                    F_l=feature,
                    F_int=feature // 2
                )
            )

            self.ups.append(
                DoubleConv(
                    feature * 2,
                    feature
                )
            )

        self.final_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        skip_connections = []

        for down in self.downs:

            x = down(x)

            skip_connections.append(x)

            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        att_idx = 0

        for idx in range(
            0,
            len(self.ups),
            2
        ):

            x = self.ups[idx](x)

            skip_connection = (
                skip_connections[idx // 2]
            )

            filtered_skip = self.att_gates[
                att_idx
            ](
                x,
                skip_connection
            )

            att_idx += 1

            if x.shape[2:] != filtered_skip.shape[2:]:

                x = TF.resize(
                    x,
                    size=filtered_skip.shape[2:]
                )

            concat_x = torch.cat(
                (
                    filtered_skip,
                    x
                ),
                dim=1
            )

            x = self.ups[idx + 1](
                concat_x
            )

        return self.final_conv(x)
