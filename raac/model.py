import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv.apply(init_weights)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class CrackDetectionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CrackDetectionModel, self).__init__()
        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128, dropout_prob=0.3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(128, 256)
        self.up1 = UpConv(256, 128)
        self.dec1 = ConvBlock(256, 128)
        self.up2 = UpConv(128, 64)
        self.dec2 = ConvBlock(128, 64)
        self.final_seg = nn.Conv2d(64, 1, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.dec1(torch.cat((self.up1(b), e2), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d1), e1), dim=1))
        seg_output = torch.sigmoid(self.final_seg(d2))
        cls_output = self.global_avg_pool(b)
        cls_output = cls_output.view(cls_output.size(0), -1)
        cls_output = self.fc(cls_output)
        return seg_output, cls_output
