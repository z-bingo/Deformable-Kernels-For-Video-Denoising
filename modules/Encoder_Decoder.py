import torch
import torch.nn as nn

class Conv_Blocks(nn.Module):
    def __init__(self, in_channel, out_channel, downsamle=False, upsample=False):
        super(Conv_Blocks, self).__init__()
        if not downsamle:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
        if upsample:
            self.layer_up = nn.Sequential(
                nn.Conv2d(out_channel, out_channel*4, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(upscale_factor=2)
            )
        else:
            self.layer_up = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.layer_up(self.layer(x))


class Encoder_Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder_Decoder, self).__init__()
        self.in_conv = Conv_Blocks(in_channel, 64, downsamle=False, upsample=False)
        self.down1 = Conv_Blocks(64, 128, downsamle=True, upsample=False)
        self.down2 = Conv_Blocks(128, 256, downsamle=True, upsample=False)
        self.down3 = Conv_Blocks(256, 512, downsamle=True, upsample=False)
        self.down_up = Conv_Blocks(512, 512, downsamle=True, upsample=True)
        self.up1 = Conv_Blocks(1024, 256, downsamle=False, upsample=True)
        self.up2 = Conv_Blocks(512, 128, downsamle=False, upsample=True)
        self.up3 = Conv_Blocks(256, 128, downsamle=False, upsample=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.out_offsets = nn.Sequential(
            nn.Conv2d(128, out_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

    def forward(self, x):
        f1 = self.in_conv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down_up(f4)
        f6 = self.up1(torch.cat([f5, f4], dim=1))
        f7 = self.up2(torch.cat([f6, f3], dim=1))
        f8 = self.up3(torch.cat([f7, f2], dim=1))
        f = self.out_conv(torch.cat([f8, f1], dim=1))
        return f, self.out_offsets(f)

if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    import torch
    writer = SummaryWriter('./logs')
    model = Encoder_Decoder(24, 21)
    print(model)
