import torch
import torch.nn as nn
from modules.Encoder_Decoder import Encoder_Decoder
from modules.DeformableConv import trilinear_interpolation, convolution_3D

class Weight_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Weight_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channel, 3, 1, 1)
        )

    def forward(self, x):
        return self.layer(x)


class Deformable_KPN(nn.Module):
    def __init__(self, burst_length=5, color=True, blind=True, sample_nums=27, groups=3):
        super(Deformable_KPN, self).__init__()
        self.depth = 3 if color else 1
        self.burst_length = burst_length
        self.blind = blind
        self.input_length = (self.burst_length + (1 if not blind else 0)) * self.depth
        self.sample_nums = sample_nums
        # split N sampled locations into s groups for avoid convergencing to a local minimum
        self.g = groups
        # offsets network
        self.encoder_decoder = Encoder_Decoder(
            in_channel=self.input_length,
            out_channel=self.sample_nums*3
        )
        # print(self.encoder_decoder)

        channel_in_weight = self.input_length + 128 + self.sample_nums*self.depth
        self.weight_subnet = Weight_Conv(
            in_channel=channel_in_weight,
            out_channel=self.sample_nums
        )

    def forward(self, data, noise_estimation=None):
        """
        We hope the shape of data is [b, c, d, h, w], and noise_estimaton is added for non-blind denoising
        :param data: [b, c, d, h, w]
        :param noise_estimation: [b, d, h, w], the estimation for middle frame
        :return:
        """
        assert len(data.size()) == 5
        b, c, d, h, w = data.size()
        if self.blind:
            net_feed = data.view(b, c*d, h, w)
        else:
            net_feed = data.view(b, c*d, h, w)
            net_feed = torch.cat([net_feed, noise_estimation.view(b, -1, h, w)], dim=1)
        # net_feed as the input of U-Net-like network
        # feature as one of the input of weight_subnet, offset for sampling the pixels
        feature, offsets = self.encoder_decoder(net_feed)
        # shape of samples [b, N, d, h, w]
        samples = trilinear_interpolation(data, offsets)
        # print('samples', samples.size())

        weight_feed = torch.cat([net_feed, feature, samples.view(b, -1, h, w)], dim=1)
        weights = self.weight_subnet(weight_feed)

        # group
        channel_per_group = self.sample_nums // self.g
        samples_group = torch.split(samples, split_size_or_sections=channel_per_group, dim=1)
        weights_group = torch.split(weights, split_size_or_sections=channel_per_group, dim=1)

        res_i = torch.zeros(b, self.g, d, h, w, device=data.device.type)
        for i in range(self.g):
            res_i[:, i, ...] = self.g * convolution_3D(samples_group[i], weights_group[i])

        # when working on eval mode, return the samples for displaying
        # self.training is inherited from nn.Module
        if self.training:
            return res_i, torch.sum(res_i, dim=1, keepdim=False)/self.g
        else:
            return res_i, torch.sum(res_i, dim=1, keepdim=False)/self.g, samples

