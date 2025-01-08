from torch import nn

from models.net0 import get_gradient
#from models.net import ESAM
import numpy as np
import torch
# from models.net import pixel_reshuffle
"""通道索引的详细解释
前 4*c 通道：表示模型的预测值。
索引范围为 0:52（如果 c = 13）。
后 4*c 通道：表示模型的梯度值。
索引范围为 52:104（如果 c = 13）。
总结
TensorFlow 和 PyTorch 版本的主要区别在于如何计算和提取通道索引。TensorFlow 版本假设梯度直接存储在固定的通道范围内，而 PyTorch 版本通过动态计算确保通道索引与实际数据一致"""

def hs2p_loss(y_true, y_pred):
    # y_true: shape (b, c, h, w)
    # y_pred: shape (b, 8*c, h, w)
    # y_pred[:,0:4c,:,:] out of each layer
    # y_pred[:,4c:8c,:,:] gradient of each layer


    # b：控制梯度损失在总损失中的权重。
    # n：分成的层数。
    # c：真实图像的通道数。
    b = 0.5 #L_st
    n = 4 #
    c = y_true.size(1)
    #y_true_grad：通过计算真实图像的梯度，获得其边缘和结构信息。
    y_true_grad = get_gradient(y_true, c)
    # esam = ESAM(c).to(y_true.device)
    # y_true_grad = esam(y_true)

    # loss：初始化总损失为0。
    # w：用于存储每层的权重。
    # sum_w：用于存储权重的总和。
    loss = 0
    w = []
    sum_w = 0

    #使用sigmoid函数的一种变形来计算每层的权重。
    for i in range(n):
        tmp_w = 1 / (1 + np.exp(2 - (i + 1)))   #1/(1+e^(2-x)) np.exp(1)
        w.append(tmp_w)
        sum_w  += tmp_w

    #归一化权重：
    for i in range(n):
        # # 获取归一化后的权重
        w_i = round(w[i] / sum_w, 2)

        #确定预测值中第 i 层的起始和结束通道索引。
        s = i * c
        e = (i + 1) * c
        ## 获取预测值的第 i 层,对应通道范围 0:c 到 3*c:4*c。
        pred_tmp = y_pred[:, s:e, :, :]

        #确定预测值中第 i 层梯度的起始和结束通道索引。
        s1 = (i + n) * c
        e1 = (i + n + 1) * c
        ## 获取预测值的第 i 层的梯度,对应通道范围 4*c:5*c 到 7*c:8*c。
        pred_grad_tmp = y_pred[:, s1:e1, :, :]

        #计算第 i 层的损失，包括像素级别的L1损失和梯度级别的L1损失，并根据权重加权累加到总损失中。
        loss += w_i * (torch.mean(torch.abs(pred_tmp - y_true)) + b * torch.mean(torch.abs(pred_grad_tmp - y_true_grad)))

    return loss
