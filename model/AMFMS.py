import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math
import pywt.data


def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)
        multi_spectral_attention_map = torch.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 output_lim,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.output_lim = output_lim
        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])

            feature = F.interpolate(feature, size=self.output_lim, mode='bilinear', align_corners=False)  # 补充操作

            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation


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


class MFAR(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv):
        super(MFAR, self).__init__()
        self.output_lim = output_lim
        self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.PReLU(num_parameters=out_size))
        self.MFMSA = MFMSAttentionBlock(in_channels=out_size, output_lim=output_lim, scale_branches=3, frequency_branches=16, frequency_selection='top', block_repetition=1)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = PixelShuffleBlock(in_size, out_size, para_upscale_factor=2)

    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.MFMSA(self.conv(torch.cat([input1, input2], 1)))


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


class SeismicRecordDownSampling_WTConv(nn.Module):
    def __init__(self, para_shot_num):
        super().__init__()
        self.dim_reduce1 = ConvBlock(para_shot_num, 8, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce2 = WTConv2d(8,8)
        self.dim_reduce3 = ConvBlock(8, 16, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce4 = WTConv2d(16,16)
        self.dim_reduce5 = ConvBlock(16, 32, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce6 = WTConv2d(32,32)

    def forward(self, x):
        dim_reduce0 = F.interpolate(x, size=[560, 70], mode='bilinear', align_corners=False)
        dim_reduce1 = self.dim_reduce1(dim_reduce0)
        dim_reduce2 = self.dim_reduce2(dim_reduce1)
        dim_reduce3 = self.dim_reduce3(dim_reduce2)
        dim_reduce4 = self.dim_reduce4(dim_reduce3)
        dim_reduce5 = self.dim_reduce5(dim_reduce4)
        dim_reduce6 = self.dim_reduce6(dim_reduce5)

        return dim_reduce6


class UNetConv2(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_is_bn):
        super(UNetConv2, self).__init__()
        if para_is_bn:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       nn.PReLU(num_parameters=para_out_size))
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       nn.PReLU(num_parameters=para_out_size))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.PReLU(num_parameters=para_out_size))
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.PReLU(num_parameters=para_out_size))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class DCEW(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_is_bn):
        super(DCEW, self).__init__()
        if para_is_bn:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       nn.PReLU(num_parameters=para_out_size))
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       nn.PReLU(num_parameters=para_out_size))
            self.WTConv2 = WTConv2d(para_out_size, para_out_size)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.PReLU(num_parameters=para_out_size))
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.PReLU(num_parameters=para_out_size))
            self.WTConv2 = WTConv2d(para_out_size, para_out_size)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.WTConv2(outputs)
        return outputs


class AMFMS_SEG(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(AMFMS_SEG, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # Encoder
        self.DCEW2 = DCEW(29, 32, self.is_batchnorm)
        self.max2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DCEW3 = DCEW(32, 64, self.is_batchnorm)
        self.max3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DCEW4 = DCEW(64, 128, self.is_batchnorm)
        self.max4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DCEW5 = DCEW(128, 256, self.is_batchnorm)
        self.max5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.center = UNetConv2(256, 512, self.is_batchnorm)

        # Decoder
        self.Up5 = MFAR(512, 256, output_lim=[50, 38], is_deconv=self.is_deconv)
        self.Up4 = MFAR(256, 128, output_lim=[100, 76], is_deconv=self.is_deconv)
        self.Up3 = MFAR(128, 64, output_lim=[200, 151], is_deconv=self.is_deconv)
        self.pixelShuffleBlock2 = PixelShuffleBlock(64, 64, para_upscale_factor=2)
        self.pixelShuffleBlock1 = PixelShuffleBlock(64, 64, para_upscale_factor=1)
        self.dc2_final = nn.Conv2d(64, 1, 1)

    def forward(self, inputs, _=None):
        '''
        :param inputs:      Input Image
        '''
        # encoder
        DCEW2 = self.DCEW2(inputs)
        max2 = self.max2(DCEW2)  # (2, 32, 200, 151)
        DCEW3 = self.DCEW3(max2)
        max3 = self.max3(DCEW3)  # (2, 64, 100, 76)
        DCEW4 = self.DCEW4(max3)
        max4 = self.max4(DCEW4)  # (2, 128, 50, 38)
        DCEW5 = self.DCEW5(max4)
        max5 = self.max5(DCEW5)  # (2, 256, 25, 19)
        center = self.center(max5)  # (2, 512, 25, 19)

        # decoder
        dc2_up5 = self.Up5(DCEW5, center)  # (2, 256, 50, 38)
        dc2_up4 = self.Up4(DCEW4, dc2_up5)  # (2, 128, 100, 76)
        dc2_up3 = self.Up3(DCEW3, dc2_up4)  # (2, 64, 200, 151)
        pixelShuffleBlock2 = self.pixelShuffleBlock2(dc2_up3)  # (2, 64, 400, 302)
        pixelShuffleBlock1 = self.pixelShuffleBlock1(pixelShuffleBlock2)  # (2, 64, 400, 302)
        dc_capture = pixelShuffleBlock1[:, :, 1:1 + 201, 1:1 + 301].contiguous()
        dc2_final = self.dc2_final(dc_capture)  # (2, 1,  201, 301)

        return dc2_final


class AMFMS(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(AMFMS, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # Encoder
        self.pre_seis_conv = SeismicRecordDownSampling_WTConv(self.in_channels)
        self.DCEW3 = DCEW(32, 64, self.is_batchnorm)
        self.max3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DCEW4 = DCEW(64, 128, self.is_batchnorm)
        self.max4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DCEW5 = DCEW(128, 256, self.is_batchnorm)
        self.max5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.center = UNetConv2(256, 512, self.is_batchnorm)

        # Decoder
        self.Up5 = MFAR(512, 256, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.Up4 = MFAR(256, 128, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.Up3 = MFAR(128, 64, output_lim=[70, 70], is_deconv=self.is_deconv)
        self.pixelShuffleBlock2 = PixelShuffleBlock(64, 64, para_upscale_factor=1)
        self.pixelShuffleBlock1 = PixelShuffleBlock(64, 64, para_upscale_factor=1)
        self.dc2_final = nn.Conv2d(64, 1, 1)

    def forward(self, inputs, _=None):
        '''
        :param inputs:      Input Image
        '''
        # encoder
        compress_seis = self.pre_seis_conv(inputs)  # (2, 32, 70, 70)
        DCEW3 = self.DCEW3(compress_seis)  # (2, 32, 400, 301)
        max3 = self.max3(DCEW3)  # (2, 64, 100, 76)
        DCEW4 = self.DCEW4(max3)  # (2, 128, 100, 76)
        max4 = self.max4(DCEW4)  # (2, 128, 50, 38)
        DCEW5 = self.DCEW5(max4)  # (2, 256, 50, 38)
        max5 = self.max5(DCEW5)  # (2, 256, 25, 19)
        center = self.center(max5)  # (2, 512, 25, 19)

        # decoder
        dc2_up5 = self.Up5(DCEW5, center)  # (2, 256, 50, 38)
        dc2_up4 = self.Up4(DCEW4, dc2_up5)  # (2, 128, 100, 76)
        dc2_up3 = self.Up3(DCEW3, dc2_up4)  # (2, 64, 200, 151)
        pixelShuffleBlock2 = self.pixelShuffleBlock2(dc2_up3)  # (2, 64, 400, 302)
        pixelShuffleBlock1 = self.pixelShuffleBlock1(pixelShuffleBlock2)  # (2, 64, 400, 302)
        dc2_final = self.dc2_final(pixelShuffleBlock1)  # (2, 1,  201, 301)
        return dc2_final


if __name__ == '__main__':
    model = AMFMS_SEG(n_classes=1, in_channels=29, is_deconv=True, is_batchnorm=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model.to(device)
    from torchsummary import summary
    summary(model, input_size=[(29, 400, 301)])

    # model = AMFMS(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
    # cuda_available = torch.cuda.is_available()
    # device = torch.device("cuda" if cuda_available else "cpu")
    # model.to(device)
    # from torchsummary import summary
    # summary(model, input_size=[(5, 1000, 70)])

