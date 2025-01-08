import torch
import torch.nn as nn
# from torchviz import make_dot
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# class GetGradient(nn.Module):
#     def __init__(self, layername=None, in_channel=13):
#         super().__init__()
#         self.layername = layername
#         self.conv_v = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
#         self.conv_h = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)

#         for param in self.conv_v.parameters():
#             param.requires_grad = False
#         for param in self.conv_h.parameters():
#             param.requires_grad = False

#         filter_v = torch.tensor([[0,-1,0],[0,0,0],[0,1,0]]).float()
#         filter_h = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]]).float()

#         self.conv_v.weight.data.copy_ = filter_v
#         self.conv_h.weight.data.copy_ = filter_h

#     def forward(self, input_l):
#         tmp_v = self.conv_v(input_l)
#         tmp_h = self.conv_h(input_l)
#         tmp_v = torch.square(tmp_v)
#         tmp_h = torch.square(tmp_h)
#         tmpvh = torch.sqrt(tmp_v + tmp_h + 1e-6)

#         return tmpvh


def get_gradient(input_l, in_channel=13):
    conv_v = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1).cuda()
    conv_h = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1).cuda()

    for param in conv_v.parameters():
        param.requires_grad = False
    for param in conv_h.parameters():
        param.requires_grad = False

    # 定义卷积核（Sobel核）。(一个水平方向、一个垂直方向)
    filter_v = torch.tensor([[0,-1,0],[0,0,0],[0,1,0]]).float().to(input_l)
    filter_h = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]]).float().to(input_l)

    conv_v.weight.data = filter_v.expand(in_channel, in_channel, 3, 3)
    conv_h.weight.data = filter_h.expand(in_channel, in_channel, 3, 3)


    # 使用卷积操作提取图像的水平梯度和垂直梯度。
    tmp_v = conv_v(input_l)
    tmp_h = conv_h(input_l)

    # 通过计算水平方向和垂直方向的梯度平方和开平方计算梯度幅值。
    tmp_v = torch.square(tmp_v)
    tmp_h = torch.square(tmp_h)
    tmpvh = torch.sqrt(tmp_v + tmp_h + 1e-6)

    return tmpvh


class Attention(nn.Module):
    def __init__(self, feature_size, reduction=256):
        #reduction: 降维因子，默认为256，用于减小模型的参数数量和计算量。
        super().__init__()
        # initial parameters
        #layer_one 和 layer_two: 这两个层是全连接层（nn.Linear），用于从降维后的特征中学习注意力权重。
        # layer_one 将通道数从 feature_size 降维到 feature_size // reduction，而 layer_two 则将其再扩展回 feature_size。
        self.layer_one = nn.Linear(feature_size, feature_size // reduction)
        self.activation_one = nn.ReLU()
        self.layer_two = nn.Linear(feature_size // reduction, feature_size)
        self.activation_two = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Hardsigmoid()

    def forward(self, input_l):
        b, c, h, w = input_l.shape
        x = input_l

        avg_pool = self.avg_pool(x)
        # 使用 reshape 后调用 contiguous，
        # 是为了确保在内存中数据是连续的，从而在后续可能涉及内存布局敏感的操作（例如使用某些 CUDA 操作）时不会出现问题。
        #这意味着对于每个样本（b批量中的每个），每个通道（c个通道）都被池化成一个1x1的空间尺寸。
        avg_pool = avg_pool.reshape(b, 1, 1, c).contiguous()
        avg_pool = self.layer_one(avg_pool)
        avg_pool = self.activation_one(avg_pool)
        avg_pool = self.layer_two(avg_pool)
        avg_pool = self.activation_two(avg_pool)

        max_pool = self.max_pool(x)
        max_pool = max_pool.reshape(b, 1, 1, c).contiguous()
        max_pool = self.layer_one(max_pool)
        max_pool = self.activation_one(max_pool)
        max_pool = self.layer_two(max_pool)
        max_pool = self.activation_two(max_pool)

        weight = self.sigmoid(max_pool + avg_pool)
        weight = weight.permute(0, 3, 1, 2).contiguous()

        return input_l * weight


class ResBlock(nn.Module):
    def __init__(self, feature_size, reduction=256):
        super().__init__()
        #conv1 和 conv2: 两个卷积层，都使用了大小为3x3的卷积核，且有填充1，这样输出特征图的空间尺寸保持不变。
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.attention = Attention(feature_size, reduction)

    def forward(self, input_l):
        x = input_l
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.attention(x)

        return input_l + x


class ResGroup(nn.Module):
    def __init__(self, feature_size, block_num=4):
        super().__init__()
        #blocks: 使用 nn.ModuleList 存储多个 ResBlock 实例。
        self.blocks = nn.ModuleList([ResBlock(feature_size) for _ in range(block_num)])
        self.conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, input_l):
        x = input_l
        for block in self.blocks:
            x = block(x)
        x = self.conv(x)

        return input_l + x


class DataFusion(nn.Module):
    def __init__(self, input_shape, feature_size):
        super().__init__()
        self.conv = nn.Conv2d(input_shape, feature_size, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.attention = Attention(feature_size)
        # 添加计数器
        self.counter = 0

    def forward(self, input_l):
        self.counter += 1
        if self.counter % 100 == 0:
            print(f"这是第 {self.counter} 次使用 DataFusion 模块")

        # print(input_l.shape)  # 确认输入的高度和宽度是否足够大
        x = self.conv(input_l)
        x = self.act(x)
        x = self.attention(x)
        return x


class BasicLayer(nn.Module):
    #feature_size: 残差组处理的特征通道数。opt_channel: 输出通道数。
    def __init__(self,feature_size, opt_channel, block_num=4):
        super().__init__()
        self.opt_channel = opt_channel
        self.resGroup = ResGroup(feature_size, block_num)
        # conv: 一个卷积层，用于将 ResGroup 的输出从 feature_size 通道转换为 opt_channel 通道。
        self.conv = nn.Conv2d(feature_size, opt_channel, kernel_size=3, padding=1)

    def forward(self, input_l):
        x = self.resGroup(input_l)
        #首先，通过一个卷积层调整特征图的通道数，然后通过计算梯度幅值来增强特征图中的边缘和纹理信息，
        feature_tmp = self.conv(x)
        feature_grad_tmp = get_gradient(feature_tmp, self.opt_channel)

        return (feature_tmp, feature_grad_tmp)


class HS2P(nn.Module):
    # def __init__(self,sar_channel, opt_channel, N=4, feature_size=256):
    def __init__(self, device, sar_channel, opt_channel, N=4, feature_size=256):
        super().__init__()
        self.opt_channel = opt_channel
        self.data_fusion = DataFusion(opt_channel + sar_channel, feature_size)
        self.blocks = nn.ModuleList([BasicLayer(feature_size, opt_channel) for _ in range(N)])
        #self.upsample = nn.ConvTranspose2d(in_channels=52, out_channels=52, kernel_size=2, stride=2)
        # self.upsample = nn.Upsample(size=(128, 128), mode='bilinear')
        # self.final_conv1x1 = nn.Conv2d(in_channels=52, out_channels=13, kernel_size=1)

    def forward(self, input_sar, input_cld):
        x = torch.cat([input_sar, input_cld], dim=1)
        x = self.data_fusion(x)
        feature = None
        feature_grad = None

        for i, block in enumerate(self.blocks):
            feature_tmp, feature_grad_tmp = block(x)

            if i == 0:
                feature = feature_tmp
                feature_grad = feature_grad_tmp
            else:
                feature = torch.cat([feature, feature_tmp], dim=1)
                feature_grad = torch.cat([feature_grad, feature_grad_tmp], dim=1)

        #随后的块的输出通过 torch.cat 方法在通道维度上累加到先前的输出上，逐步构建一个更丰富的特征和梯度表示。
        X = torch.cat([feature, feature_grad], dim=1)
        # X = feature
        # X = self.upsample(X)
        # X = self.final_conv1x1(X)

        # return (X[:, 3*self.opt_channel : 4*self.opt_channel, :, :], X)
        return X


if __name__ == '__main__':
    # cld = torch.randn(size=(2, 13, 128, 128))
    # sar = torch.randn(size=(2, 2, 128, 128))
    # # opt = torch.randn(size=(2, 13, 128, 128))
    # model = HS2P(13, 2, 4)
    # y = model(cld, sar)
    # print(y.shape) # (2, 104, 128, 128)

    cld = torch.randn(size=(2, 13, 128, 128))
    sar = torch.randn(size=(2, 2, 128, 128))
    # opt = torch.randn(size=(2, 13, 128, 128))
    model = HS2P(2, 2, 13,4)
    print(model)
    # y = model(cld, sar)
    y = model(sar,cld)
    y = y[0]
    print(y.shape)  # (2, 3, 128, 128)
    # print(y1.shape)
