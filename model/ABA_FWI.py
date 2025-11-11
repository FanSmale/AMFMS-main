import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape

            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.PReLU())
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.PReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        y = self.fc(avg_pool) + self.fc(max_pool)
        y = torch.sigmoid(y)
        return x * y


class SpatialAttention1(nn.Module):
    def __init__(self):
        super(SpatialAttention1, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        # import matplotlib.pyplot as plt
        # plt.imshow(x.detach().cpu().numpy()[102][15][:][:])
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y

class CBAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMModule, self).__init__()
        # self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        #  x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ABA_FWI(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_FWI, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.CBAM_model1 = CBAMModule(256)
        self.CBAM_model2 = CBAMModule(128)
        self.CBAM_model3 = CBAMModule(64)
        self.CBAM_model4 = CBAMModule(32)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

        self.wt0 = WTConv2d(128,128)
        self.wt1 = WTConv2d(128,128)
        self.wt2 = WTConv2d(256,256)

    def forward(self, x):
        # Encoder Part
        x0 = self.convblock1(x)  # (None, 32, 500, 70)
        x0 = self.convblock2_1(x0)  # (None, 64, 250, 70)
        x0 = self.convblock2_2(x0)  # (None, 64, 250, 70)
        x0 = self.convblock3_1(x0)  # (None, 64, 125, 70)
        x0 = self.convblock3_2(x0)  # (None, 64, 125, 70)

        x1 = self.convblock4_1(x0)  # (None, 128, 63, 70)
        x2 = self.convblock4_2(x1)  # (None, 128, 63, 70)
        x2_wt = self.wt0(x2)

        x3 = self.convblock5_1(x2_wt)  # (None, 128, 40, 40)
        x4 = self.convblock5_2(x3)  # (None, 128, 40, 40)
        x4_wt = self.wt1(x4)

        x5 = self.convblock6_1(x4_wt)  # (None, 256, 20, 20)
        x6 = self.convblock6_2(x5)  # (None, 256, 20, 20)
        x6_wt = self.wt2(x6)

        x7 = self.convblock7_1(x6_wt)  # (None, 256, 10, 10)
        x8 = self.convblock7_2(x7)  # (None, 256, 10, 10)

        x9 = self.convblock8_1(x8)  # (None, 512, 5, 5)
        x10 = self.convblock8_2(x9)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y0 = self.deconv2_1(x10)  # (None, 256, 10, 10)
        y0_concat = torch.cat((x8, y0), dim=1)
        y0_concat = self.change_channel1(y0_concat)
        y1 = self.deconv2_2(y0_concat)  # (None, 256, 10, 10)
        y1_ca = self.CBAM_model1(y1)

        y2 = self.deconv3_1(y1_ca)  # (None, 128, 20, 20)
        x6_wt = self.change_channel0(x6_wt)
        y2_concat = torch.cat((x6_wt, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)
        y3_ca = self.CBAM_model2(y3)

        y4 = self.deconv4_1(y3_ca)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)
        y5_ca = self.CBAM_model3(y5)

        y6 = self.deconv5_1(y5_ca)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)
        y7_ca = self.CBAM_model4(y7)

        # pain_openfwi_velocity_model(y7_ca[0,0,:,:].cpu().detach().numpy())
        y8 = F.pad(y7_ca, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9


class ABA_Loss(nn.Module):
    """
    The ablation experiment.
    Add skip connections into InversionNet.
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_Loss, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 40, 40)
        x = self.convblock5_2(x)  # (None, 128, 40, 40)
        x = self.convblock6_1(x)  # (None, 256, 20, 20)
        x = self.convblock6_2(x)  # (None, 256, 20, 20)
        x1 = x  # (None, 64, 20, 20)

        x = self.convblock7_1(x)  # (None, 256, 10, 10)
        x = self.convblock7_2(x)  # (None, 256, 10, 10)
        x2 = x  # (None, 64, 20, 20)

        x = self.convblock8_1(x)  # (None, 512, 5, 5)
        x = self.convblock8_2(x)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y = self.deconv2_1(x)  # (None, 256, 10, 10)
        y_concat = torch.cat((x2, y), dim=1)
        y_concat = self.change_channel1(y_concat)
        y1 = self.deconv2_2(y_concat)  # (None, 256, 10, 10)
        y2 = self.deconv3_1(y1)  # (None, 128, 20, 20)
        x1 = self.change_channel0(x1)
        y2_concat = torch.cat((x1, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)
        y4 = self.deconv4_1(y3)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)
        y6 = self.deconv5_1(y5)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)
        y8 = F.pad(y7, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9


if __name__ == '__main__':
    input = torch.rand((5, 5, 1000, 70))

    model = ABA_FWI()
    output = model(input)
    print('ok')
