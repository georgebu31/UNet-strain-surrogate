import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, 3, padding=1)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(1024, 512, 1))
        self.conv13 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv14 = nn.Conv2d(512, 512, 3, padding=1)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(512, 256, 1))
        self.conv15 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256, 128, 1))
        self.conv17 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv18 = nn.Conv2d(128, 128, 3, padding=1)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128, 64, 1))
        self.conv19 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv20 = nn.Conv2d(64, 64, 3, padding=1)

        self.up5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(64, 32, 1))
        self.conv21 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv22 = nn.Conv2d(32, 32, 3, padding=1)

        self.act = nn.ReLU()
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        skip1 = x
        x = self.pool1(x)

        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        skip2 = x
        x = self.pool2(x)

        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        skip3 = x
        x = self.pool3(x)

        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x))
        skip4 = x
        x = self.pool4(x)

        x = self.act(self.conv9(x))
        x = self.act(self.conv10(x))
        skip5 = x
        x = self.pool5(x)

        x = self.act(self.conv11(x))
        x = self.act(self.conv12(x))

        x = self.up1(x)
        skip5 = skip5[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip5], dim=1)
        x = self.act(self.conv13(x))
        x = self.act(self.conv14(x))

        x = self.up2(x)
        skip4 = skip4[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip4], dim=1)
        x = self.act(self.conv15(x))
        x = self.act(self.conv16(x))

        x = self.up3(x)
        skip3 = skip3[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip3], dim=1)
        x = self.act(self.conv17(x))
        x = self.act(self.conv18(x))

        x = self.up4(x)
        skip2 = skip2[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip2], dim=1)
        x = self.act(self.conv19(x))
        x = self.act(self.conv20(x))

        x = self.up5(x)
        skip1 = skip1[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip1], dim=1)
        x = self.act(self.conv21(x))
        x = self.act(self.conv22(x))

        return self.head(x)
