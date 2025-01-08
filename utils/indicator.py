
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
#numpy版本
def get_sam_np(y_true, y_predict):
    epsilon = 1e-10  # 添加一个非常小的值以避免除以零
    mat = np.multiply(y_true, y_predict)
    mat = np.sum(mat, axis=1)
    mat = np.divide(mat, np.sqrt(np.sum(np.multiply(y_true, y_true), axis=1)) + epsilon)
    mat = np.divide(mat, np.sqrt(np.sum(np.multiply(y_predict, y_predict), axis=1)) + epsilon)
    mat = np.arccos(np.clip(mat, -1, 1))
    return mat
def cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image using NumPy."""
    mat = get_sam_np(y_true, y_predict)
    return np.mean(mat)

#张量版本
# def get_sam_torch(y_true, y_predict):
#     epsilon = 1e-8  # 避免除以零
#     dot_product = torch.sum(y_true * y_predict, dim=1)
#     norm_y_true = torch.sqrt(torch.sum(y_true * y_true, dim=1) + epsilon)
#     norm_y_predict = torch.sqrt(torch.sum(y_predict * y_predict, dim=1) + epsilon)
#     cos_similarity = dot_product / (norm_y_true * norm_y_predict)
#     cos_similarity = torch.clamp(cos_similarity, -1 + epsilon, 1 - epsilon)  # 确保在[-1, 1]范围内，考虑浮点误差
#     sam_angles = torch.acos(cos_similarity)
#     return sam_angles
#
# def cloud_mean_sam(y_true, y_predict):
#     """Computes the SAM over the full image using PyTorch."""
#     sam_angles = get_sam_torch(y_true, y_predict)
#     # 计算所有角度的平均值
#     mean_sam = torch.mean(sam_angles)
#     return mean_sam

def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE using PyTorch."""
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae

def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE using PyTorch."""
    mse = torch.mean((y_true - y_pred) ** 2)
    return mse

def RMSE(y_true, y_pred):
    """Computes the RMSE using PyTorch."""
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def gaussian(window_size, sigma):
    #计算高斯窗口的每个元素的值。它是高斯分布的概率密度函数的一部分，根据距离窗口中心的距离 x - window_size / 2 计算每个点的权重。
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    #将高斯窗口的所有值归一化，使其总和等于1，以便用于加权计算。
    return gauss / gauss.sum()

#个函数创建了一个用于 SSIM 计算的窗口。它首先生成一个一维的高斯窗口，然后通过外积得到二维窗口。这个窗口会被用来对图像进行加权，以计算 SSIM。
def create_window(window_size, channel):
    #调用 gaussian 函数以创建一个一维的高斯窗口，其中 window_size 是窗口的大小，1.5 是高斯分布的标准差。unsqueeze(1)将一维的高斯窗口变为二维，以便进行矩阵运算。
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    #进行一维高斯窗口的外积运算，得到一个二维的高斯窗口。这个操作将一维窗口在水平和垂直方向进行扩展，以生成一个平面的二维窗口。
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #将二维的高斯窗口扩展为指定通道数的四维张量，并将其包装为 PyTorch 的 Variable 对象。这个窗口将用于图像的结构相似性计算。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2,device):
    #获取图像的大小，其中 channel 表示图像的通道数。
    (_, channel, _, _) = img1.size()
    #定义了用于计算 SSIM 的窗口大小。
    window_size = 11
    window = create_window(window_size, channel).to(device)
    #接下来，计算了两幅图像的均值 mu1 和 mu2，均值的计算使用了卷积操作，以窗口为基础进行局部平均计算。
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def PSNR(img1, img2, mask=None):
    # mask
    # mask是可选参数，用于指定一个掩码图像，如果提供，将根据掩码图像计算 PSNR。
    # 如果没有提供掩码图像，将计算两幅图像的均方误差（MSE）。
    if mask is not None:
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)

        # 添加一个非常小的数值eps，以避免除以零的错误
    eps = 1e-10
    # 如果 MSE 为零，表示两幅图像完全相同，此时返回 PSNR 值为100。
    # 否则，根据 MSE 的计算结果，使用固定的峰值范围（PIXEL_MAX = 1）计算 PSNR，并返回 PSNR 值。
    mse += eps
    if mse < eps:  # 这里检查mse是否非常接近零
        return 100
    PIXEL_MAX = 1
    try:
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    except ValueError as e:
        print(f"Error calculating PSNR: {e}")
        psnr = 0  # 或者其他合适的默认值

    return psnr
