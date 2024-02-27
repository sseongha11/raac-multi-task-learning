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
            nn.LeakyReLU(0.01, inplace=True),
            # # Adding another convolutional layer to deepen the block
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.01, inplace=True)
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


class CrackDetectionModelDeep1(nn.Module):
    def __init__(self, num_classes=3):
        super(CrackDetectionModelDeep1, self).__init__()
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128, dropout_prob=0.3)
        self.enc3 = ConvBlock(128, 256)  # Additional encoder block
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Additional pooling layer
        self.bottleneck = ConvBlock(256, 512)  # Increased bottleneck depth
        self.up1 = UpConv(512, 256)
        self.dec1 = ConvBlock(512, 256)  # Adjusted for new up1 output
        self.up2 = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.up3 = UpConv(128, 64)  # Additional up-convolution
        self.dec3 = ConvBlock(128, 64)  # Additional decoder block
        self.final_seg = nn.Conv2d(64, 1, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # Adjusted for new bottleneck features

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        d1 = self.dec1(torch.cat((self.up1(b), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d1), e2), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d2), e1), dim=1))  # Incorporate the additional decoder block
        seg_output = torch.sigmoid(self.final_seg(d3))
        cls_output = self.global_avg_pool(b)
        cls_output = cls_output.view(cls_output.size(0), -1)
        cls_output = self.fc(cls_output)
        return seg_output, cls_output
