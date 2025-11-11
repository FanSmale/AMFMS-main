from torch import autograd
from data.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

l1loss = nn.L1Loss()
l2loss = nn.MSELoss()


def loss_tv_1p_edge_ref_w(pred, vmodels, edges):
    """
    求两图像在两个方向上偏微分的一阶导数   加反射系数权重
    :param pred:
    :param vmodel_ideal:
    :return:
    """
    pred_x, pred_y = total_variation_loss_xy(pred)
    vmodel_ideal_x, vmodel_ideal_y = total_variation_loss_xy(vmodels)
    total_variation = torch.abs(pred_x - vmodel_ideal_x) + torch.abs(pred_y - vmodel_ideal_y)
    edge_weight = dilate_tv(edges)

    ref = reflection_coe(vmodels)
    ref_weight = reflection_weight(ref, edges)
    ref_variation = total_variation * ref_weight

    loss = torch.sum(ref_variation)
    loss = loss / (vmodels.size(0) * torch.sum(edge_weight))
    return loss


def loss_tv1(pred, vmodels, edges):
    """
    求两图像在两个方向上偏微分的一阶导数   加反射系数权重
    :param pred:
    :param vmodel_ideal:
    :return:
    """
    pred_x, pred_y = total_variation_loss_xy(pred)
    vmodel_ideal_x, vmodel_ideal_y = total_variation_loss_xy(vmodels)
    total_variation = torch.abs(pred_x - vmodel_ideal_x) + torch.abs(pred_y - vmodel_ideal_y)
    edge_weight = dilate_tv(edges)

    ref = reflection_coe(vmodels)
    ref_weight = reflection_weight(ref, edges)
    ref_variation = total_variation * ref_weight

    loss = torch.sum(ref_variation)
    loss = loss / (vmodels.size(0) * torch.sum(edge_weight))
    return loss


def dilate_tv(loss_out_w):
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float).to('cuda')
    loss_out_w = loss_out_w.to(torch.float)
    dilated_tensor = F.conv2d(loss_out_w, kernel,
                              padding=1, stride=1)
    result = torch.zeros_like(dilated_tensor)
    result[dilated_tensor != 0 ] = 1
    return result


def reflection_coe(vmodels):
    """
    计算速度模型的反射系数
    """
    x_deltas = vmodels[:, :, 1:, :] - vmodels[:, :, :-1, :]
    x_sum = vmodels[:, :, 1:, :] + vmodels[:, :, :-1, :]
    ref = x_deltas / x_sum

    ref[torch.isnan(ref)] = 0 # 归一化后速度值为0，去除除数为0的情况
    result = torch.zeros_like(vmodels)

    # 将原始矩阵放入全零矩阵中
    result[:, :, 1:, :] = torch.abs(ref)
    return result


def reflection_weight(ref, edges):
    """
    基于反射数据的权重系数
    :param ref:
    :return:
    """
    # 定义一个3x3的最大池化层
    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    # 计算邻域内的最大值
    ref_max = max_pool(ref)
    edge_dilate = dilate_tv(edges)
    ref_ideal = ref_max * edge_dilate
    ref_result = torch.where((ref_ideal != 0) & (ref_ideal < 0.05), 2, 1) * edge_dilate

    return ref_result


def total_variation_loss_xy(vmodel_ideal):
    """
    :param vmodel_ideal:   vmodels  tensor  [none, 1, 70, 70]
    :return: tensor  [none, 1, 70, 70]
    """
    # 计算图像在 x 和 y 方向的梯度
    x_deltas = vmodel_ideal[:, :, 1:, :] - vmodel_ideal[:, :, :-1, :]
    y_deltas = vmodel_ideal[:, :, :, 1:] - vmodel_ideal[:, :, :, :-1]

    x_deltas_padded_matrix = torch.zeros_like(vmodel_ideal)
    y_deltas_padded_matrix = torch.zeros_like(vmodel_ideal)

    # 将原始矩阵放入全零矩阵中
    x_deltas_padded_matrix[:, :, 1:, :] = x_deltas
    y_deltas_padded_matrix[:, :, :, 1:] = y_deltas

    return x_deltas_padded_matrix, y_deltas_padded_matrix


def calculate_edge(edges):
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float).to('cuda')
    loss_out_w = edges.to(torch.float)
    dilated_tensor = F.conv2d(loss_out_w, kernel, padding=1, stride=1)
    result = torch.zeros_like(dilated_tensor)
    result[dilated_tensor != 0] = 1
    return result


def criterion_inv(outputs, vmodels):
    loss_g2v = l2loss(outputs, vmodels)
    return loss_g2v


def criterion_fcn(outputs, vmodels):
    loss_g1v = l1loss(outputs, vmodels)
    return loss_g1v


def criterion_MMT_SEG(pred, gt, lambda_tv=0.001, epsilon=1e-6):
    # 基础损失项
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    # 原始L1/L2损失
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    base_loss = loss_g1v + loss_g2v

    # 计算TV正则化项（各向同性）
    # 假设pred形状为 (batch, channels, height, width)
    if pred.dim() == 4:
        # 计算水平和垂直差分
        diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # 水平方向差分 (height, width-1)
        diff_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # 垂直方向差分 (height-1, width)

        # 为了保持张量的维度一致，对差分进行填充
        # diff_x需要在最后一个维度（宽度）上填充一个边界
        # diff_y需要在倒数第二个维度（高度）上填充一个边界
        diff_x = F.pad(diff_x, (0, 1, 0, 0))  # (0, 1)表示在宽度方向右侧填充1列，(0, 0)表示在高度方向不填充
        diff_y = F.pad(diff_y, (0, 0, 1, 0))  # (0, 0)表示在宽度方向不填充，(1, 0)表示在高度方向底部填充1行
    else:
        raise ValueError("Unsupported input shape")

    # 各向同性TV计算
    tv_loss = torch.sqrt(diff_x.pow(2) + diff_y.pow(2) + epsilon).sum()  # 加epsilon防止零梯度
    tv_loss = tv_loss / pred.size(0)  # 对所有样本和通道取平均

    # 总损失
    total_loss = base_loss + lambda_tv * tv_loss
    return total_loss, base_loss, tv_loss


def criterion_MMT_Open(pred, gt, lambda_tv=0.00001, epsilon=1e-6):
    # 基础损失项
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    # 原始L1/L2损失
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    base_loss = loss_g1v + loss_g2v

    # 计算TV正则化项（各向同性）
    # 假设pred形状为 (batch, channels, height, width)
    if pred.dim() == 4:
        # 计算水平和垂直差分
        diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # 水平方向差分 (height, width-1)
        diff_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # 垂直方向差分 (height-1, width)

        # 为了保持张量的维度一致，对差分进行填充
        # diff_x需要在最后一个维度（宽度）上填充一个边界
        # diff_y需要在倒数第二个维度（高度）上填充一个边界
        diff_x = F.pad(diff_x, (0, 1, 0, 0))  # (0, 1)表示在宽度方向右侧填充1列，(0, 0)表示在高度方向不填充
        diff_y = F.pad(diff_y, (0, 0, 1, 0))  # (0, 0)表示在宽度方向不填充，(1, 0)表示在高度方向底部填充1行
    else:
        raise ValueError("Unsupported input shape")

    # 各向同性TV计算
    tv_loss = torch.sqrt(diff_x.pow(2) + diff_y.pow(2) + epsilon).sum()  # 加epsilon防止零梯度
    tv_loss = tv_loss / pred.size(0)  # 对所有样本和通道取平均

    # 总损失
    total_loss = base_loss + lambda_tv * tv_loss

    return total_loss, base_loss, tv_loss


class LossDDNet:
    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        '''
        Define the loss function of DDNet
        :param weights:         The weights of the two decoders in the calculation of the loss value.
        :param entropy_weight:  The weights of the two output channels in the second decoder.
        '''

        self.criterion1 = nn.MSELoss()
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight=ew)    # For multi-classification, the current issue is a binary problem (either black or white).
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        '''
        :param outputs1: Output of the first decoder
        :param outputs2: Velocity model
        :param targets1: Output of the second decoder
        :param targets2: Profile of the speed model
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())
        criterion = (self.weights[0] * mse + self.weights[1] * cross)

        return criterion


class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


def criterion_g(pred, gt, net_d=None):
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = 100 * loss_g1v + 100 * loss_g2v
    if net_d is not None:
        loss_adv = -torch.mean(net_d(pred))
        loss += loss_adv
    return loss, loss_g1v, loss_g2v
