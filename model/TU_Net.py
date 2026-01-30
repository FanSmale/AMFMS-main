# -*- coding: utf-8 -*-
"""
--------------

@Time：2024/7/1 21:11

@Author: wangyumei

"""
from model.dcn.TWM import *

class ConvBlock(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_kernel_size=(3, 3), para_stride=(1, 1),
                 para_padding=(1, 1), para_is_bn=True, para_active_function=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        if para_is_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(para_in_size, para_out_size, para_kernel_size, para_stride, para_padding),
                nn.BatchNorm2d(para_out_size),
                para_active_function)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(para_in_size, para_out_size, para_kernel_size, para_stride, para_padding),
                para_active_function)
    def forward(self, x):
        return self.conv(x)


class SeismicRecordDownSampling(nn.Module):
    def __init__(self, para_shot_num):
        super().__init__()
        self.dim_reduce1 = ConvBlock(para_shot_num, 8, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce2 = ConvBlock(8, 8, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=(1, 1))
        self.dim_reduce3 = ConvBlock(8, 16, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce4 = ConvBlock(16, 16, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=(1, 1))
        self.dim_reduce5 = ConvBlock(16, 32, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce6 = ConvBlock(32, 32, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=(1, 1))
    def forward(self, x):
        dim_reduce0 = F.interpolate(x, size=[560, 70], mode='bilinear', align_corners=False) # (1, 5, 560, 70)
        dim_reduce1 = self.dim_reduce1(dim_reduce0) # (1, 8, 280, 70)
        dim_reduce2 = self.dim_reduce2(dim_reduce1) # (1, 8, 280, 70)
        dim_reduce3 = self.dim_reduce3(dim_reduce2) # (1, 16, 140, 70)
        dim_reduce4 = self.dim_reduce4(dim_reduce3) # (1, 16, 140, 70)
        dim_reduce5 = self.dim_reduce5(dim_reduce4) # (1, 32, 70, 70)
        dim_reduce6 = self.dim_reduce6(dim_reduce5) # (1, 32, 70, 70)
        return dim_reduce6


class UNetConv2(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_is_bn, para_active_func=nn.ReLU(inplace=True)):
        super(UNetConv2, self).__init__()
        if para_is_bn:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       para_active_func)
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       para_active_func)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       para_active_func)
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       para_active_func)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUp1(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv=True):
        super(UNetUp1, self).__init__()
        self.output_lim = output_lim
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, input):
        input = self.up(input)
        output = F.interpolate(input, size=self.output_lim, mode='bilinear', align_corners=False)
        return output


class UNetUp2(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv, active_function=nn.ReLU(inplace=True)):
        super(UNetUp2, self).__init__()
        self.output_lim = output_lim
        self.conv = UNetConv2(in_size, out_size, True, active_function)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = PixelShuffleBlock(in_size, out_size, para_upscale_factor=2)
    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))


class PixelShuffle(nn.Module):
    def __init__(self, para_upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = para_upscale_factor
    def forward(self, inputs):
        channels, height, width = inputs.size()
        new_channels = channels // (self.upscale_factor ** 2)
        new_height, new_width = height * self.upscale_factor, width * self.upscale_factor
        inputs = inputs.view(new_channels, self.upscale_factor, self.upscale_factor, height, width)
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        outputs = inputs.view(new_channels, new_height, new_width)
        return outputs


class PixelShuffleBlock(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_upscale_factor, para_kernel_size=(3, 3),
                 para_stride=(1, 1), para_padding=(1, 1)):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(para_in_size, para_out_size * para_upscale_factor ** 2,
                              para_kernel_size, para_stride, para_padding)
        self.ps = nn.PixelShuffle(para_upscale_factor)
    def forward(self, inputs):
        outputs = self.ps(self.conv(inputs))
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, para_in_size, para_out_size, use_1x1conv=False, para_stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(para_in_size, para_out_size, kernel_size=(3, 3), stride=para_stride, padding=1)
        self.conv2 = nn.Conv2d(para_out_size, para_out_size, kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                para_in_size, para_out_size, kernel_size=(1, 1), stride=para_stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(para_out_size)
        self.bn2 = nn.BatchNorm2d(para_out_size)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


class ConvBlockTanh(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=1):
        super(ConvBlockTanh, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, para_kernel_size, para_stride, para_padding),
                                  nn.BatchNorm2d(para_out_size),
                                  nn.Tanh())
    def forward(self, x):
        return self.conv(x)


class TU_Net(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(TU_Net, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # Encoder
        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)
        self.down3 = UNetConv2(32, 64, self.is_batchnorm)
        self.max3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.down4 = UNetConv2(64, 128, self.is_batchnorm)
        self.max4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.down5 = UNetConv2(128, 256, self.is_batchnorm)
        self.max5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.center = UNetConv2(256, 512, self.is_batchnorm)

        # TWM
        self.TWM2 = TextureWarpingModule(256, 32, 4, 4, 0)
        self.TWM3 = TextureWarpingModule(128, 32, 2, 4, 0)
        self.TWM4 = TextureWarpingModule(64, 32, 1, 4, 0)

        # Decoder
        self.Up5 = UNetUp2(512, 256, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.Up4 = UNetUp2(256, 128, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.Up3 = UNetUp2(128, 64, output_lim=[70, 70], is_deconv=self.is_deconv)
        self.pixelShuffleBlock = PixelShuffleBlock(64, 64, para_upscale_factor=1)
        self.residual = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.residua2 = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.residua3 = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.dc2_final = ConvBlockTanh(64, 1)

    def forward(self, inputs, _ = None):
        '''
        :param inputs:      Input Image
        '''
        compress_seis = self.pre_seis_conv(inputs)  # (2, 32, 70, 70)

        down3 = self.down3(compress_seis)  # (2, 64, 70, 70)
        max3 = self.max3(down3)  # (2, 64, 35, 35)
        down4 = self.down4(max3)  # (2, 128, 35, 35)
        max4 = self.max4(down4)  # (2, 128, 18, 18)
        down5 = self.down5(max4)  # (2, 256, 18, 18)
        max5 = self.max5(down5)  # (2, 256, 9, 9)
        center = self.center(max5)  # (2, 512, 9, 9)

        # TWM
        TWM2 = self.TWM2(down5, compress_seis)  # (2, 256, 18, 18)
        TWM3 = self.TWM3(down4, compress_seis)  # (2, 128, 35, 35)
        TWM4 = self.TWM4(down3, compress_seis)  # (2, 64, 70, 70)

        # Decoder
        dc2_up5 = self.Up5(TWM2[0], center)  # (2, 256, 18, 18)
        dc2_up4 = self.Up4(TWM3[0], dc2_up5)  # (2, 128, 35, 35)
        dc2_up3 = self.Up3(TWM4[0], dc2_up4)  # (2, 64, 70, 70)
        pixelShuffleBlock = self.pixelShuffleBlock(dc2_up3)  # (2, 64, 70, 70)
        dc_residual1 = self.residual(pixelShuffleBlock)  # (2, 64, 70, 70)
        dc_residual2 = self.residua2(dc_residual1)  # (2, 64, 70, 70)
        dc_residual3 = self.residua3(dc_residual2)  # (2, 64, 70, 70)
        dc2_final = self.dc2_final(dc_residual3)  # (2, 1, 70, 70)
        return dc2_final


if __name__ == '__main__':
    model = TU_Net(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model.to(device)
    from torchsummary import summary
    summary(model, input_size=[(5, 1000, 70)])
