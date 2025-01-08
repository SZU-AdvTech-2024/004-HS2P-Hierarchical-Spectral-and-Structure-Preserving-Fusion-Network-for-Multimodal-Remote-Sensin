import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import lpips

import skimage.measure
from sewar.full_ref import vifp



def gaussian(window_size, sigma):
    # torch.cuda.set_device(6)

    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    # torch.cuda.set_device(6)

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2,device):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).to(device)
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

def PSNR(img1, img2):
    # torch.cuda.set_device(6)


    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def RMSE(img1, img2):
    """
    计算均方根误差（RMSE）

    参数:
        img1: torch.Tensor, 第一个图像
        img2: torch.Tensor, 第二个图像

    返回:
        float, RMSE 值
    """
    # 计算均方误差（MSE）
    mse = torch.mean((img1 - img2) ** 2)

    # 计算均方根误差（RMSE）
    rmse = math.sqrt(mse.item())

    return rmse


def get_sam(y_true, y_predict):
    """Computes the SAM array."""
    mat = torch.mul(y_true, y_predict)
    mat = torch.sum(mat, dim=1)
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.mul(y_true, y_true), dim=1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.mul(y_predict, y_predict), dim=1)))
    mat = torch.acos(torch.clamp(mat, -1, 1))

    return mat


def cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image."""
    mat = get_sam(y_true, y_predict)

    return torch.mean(mat)


def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return torch.mean(torch.abs(y_pred - y_true))


def compute_CC(pred, truth):
    # 计算两个张量的均值
    mean1 = torch.mean(pred)
    mean2 = torch.mean(truth)

    # 计算两个张量的标准差
    std1 = torch.std(pred)
    std2 = torch.std(truth)

    # 计算相关性系数
    return torch.mean((pred - mean1) * (truth - mean2) / (std1 * std2))


def cloud_cover_PSNR(mask, pred, truth):
    return PSNR((1-mask) * pred, (1-mask) * truth)

def cloud_cover_SSIM(mask, pred, truth):
    return SSIM(((1-mask) * pred).to(torch.float32), ((1-mask) * truth).to(torch.float32))

def cloud_cover_MAE(mask, pred, truth):
    return cloud_mean_absolute_error((1-mask) * pred, (1-mask) * truth)

def cloud_cover_SAM(mask, pred, truth):
    return cloud_mean_sam((1-mask) * pred, (1-mask) * truth)

def compute_cloud_cover_CC(mask, pred, truth):

    pred = pred * (1-mask)
    truth = truth * (1-mask)

    # 计算两个张量的均值
    mean1 = torch.mean(pred)
    mean2 = torch.mean(truth)

    # 计算两个张量的标准差
    std1 = torch.std(pred)
    std2 = torch.std(truth)

    # 计算相关性系数
    return torch.mean((pred - mean1) * (truth - mean2) / (std1 * std2))


def compute_LPIPS(pred, truth,device):
    lpips_model = lpips.LPIPS(net="alex").to(device)
    distance = lpips_model(pred[:, 1:4, :, :], truth[:, 1:4, :, :])

    return distance.item()


def compute_en(image):
    image = image.squeeze()[3,:,:]
    entropy_value = skimage.measure.shannon_entropy(image.detach().cpu().numpy(), base=2)
    
    return entropy_value


def compute_vif(img1, img2):
    img1 = img1.squeeze()[3,:,:].detach().cpu().numpy()
    img2 = img2.squeeze()[3,:,:].detach().cpu().numpy()

    vif = vifp(img1, img2)
    return vif


def compute_sd(image):
    mean = torch.mean(image)
    
    # 计算标准差
    sd = torch.sqrt(torch.mean((image - mean) ** 2))
    
    return sd

def avgGradient(image):
    image = image.squeeze().mean(dim=0).detach().cpu().numpy()
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(width):
	    for j in range(heigt):
		    dx = float(image[i,j+1])-float(image[i,j])
		    dy = float(image[i+1,j])-float(image[i,j])
		    ds = math.sqrt((dx*dx+dy*dy)/2)
		    tmp += ds
    
    imageAG = tmp/(width*heigt)
    return imageAG


# Dice系数
def dice_coeff(pred, target):
    result = 0

    smooth = 1.
    for i, j in zip(pred, target):
        m1 = i.view(i.size(0), -1)  # Flatten
        m2 = j.view(j.size(0), -1)  # Flatten
        intersection = (m1 * m2).sum()

        result += (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
 
    return result / 3


def pcc(x, y):
    """
    计算两个张量之间的皮尔逊相关系数
    :param x: 张量 x
    :param y: 张量 y
    :return: 相关系数
    """
    # 确保输入是1D张量
    x = x.view(-1)
    y = y.contiguous().view(-1)


    #y = y.view(-1)

    # x = x.reshape(-1)
    # y = y.reshape(-1)

    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 计算差异
    diff_x = x - mean_x
    diff_y = y - mean_y

    # 计算相关系数
    numerator = torch.sum(diff_x * diff_y)
    denominator = torch.sqrt(torch.sum(diff_x ** 2) * torch.sum(diff_y ** 2))
    correlation = numerator / denominator

    return correlation