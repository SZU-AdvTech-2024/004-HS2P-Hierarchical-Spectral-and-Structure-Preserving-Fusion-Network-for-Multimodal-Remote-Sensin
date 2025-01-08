import os
import numpy as np

import argparse
import random
import rasterio
import csv

import torch
from torch.utils.data import Dataset

from utils.feature_detectors import get_cloud_cloudshadow_mask

class AlignedDataset(Dataset):

    def __init__(self, opts, filelist):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # 初始化函数中设置了一些数据的处理参数，如 clip_min 和 clip_max 表示对不同类型数据的裁剪范围，max_val 表示最大值，以及 scale 表示缩放因子。
        # 数据集包含了SAR图像、云层遮挡的无云层图像和有云层图像。
        self.opts = opts

        self.filelist = filelist
        self.n_images = len(self.filelist)
        #这个范围定义是为了在数据归一化过程中限制通道值的范围，以确保数据在一定的可控范围内。
        self.clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                    [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000

    #用于获取数据集中的一项数据。
    def __getitem__(self, index):
        #根据索引 index 获取文件ID，然后构建SAR图像路径、无云层图像路径和有云层图像路径。
        fileID = self.filelist[index]
        s1_path = os.path.join(self.opts.input_data_folder, fileID[1], fileID[4])
        s2_cloudfree_path = os.path.join(self.opts.input_data_folder, fileID[2], fileID[4])
        s2_cloudy_path = os.path.join(self.opts.input_data_folder, fileID[3], fileID[4])
        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
        s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')

        #检查是否要使用云掩膜,如果是，将调用函数，将云掩膜中不等于0的像素值设置为1，以得到一个二进制云掩膜。
        if self.opts.is_use_cloudmask:
            cloud_mask = get_cloud_cloudshadow_mask(s2_cloudy_data, self.opts.cloud_threshold)
            cloud_mask[cloud_mask != 0] = 1
        '''
        for SAR, clip param: [-25.0, -32.5], [0, 0]
                 minus the lower boundary to be converted to positive
                 normalized by clip_max - clip_min, and increase by max_val
        for optical, clip param: 0, 10000
                     normalized by scale
        '''
        #函数被用于将图像数据进行标准化处理，每种类型的数据都会按照不同的方式标准化。
        s1_data = self.get_normalized_data(s1_data, data_type=1)
        s2_cloudfree_data = self.get_normalized_data(s2_cloudfree_data, data_type=2)
        s2_cloudy_data = self.get_normalized_data(s2_cloudy_data, data_type=3)
        #对标准化后的数据进行处理后，将其转换为 PyTorch 张量对象，以便后续深度学习模型的输入。
        s1_data = torch.from_numpy(s1_data)
        s2_cloudfree_data = torch.from_numpy(s2_cloudfree_data)
        s2_cloudy_data = torch.from_numpy(s2_cloudy_data)
        #如果 self.opts.is_use_cloudmask 为真，那么还会对云掩膜 cloud_mask 进行相同的处理，将其转换为 PyTorch 张量。
        if self.opts.is_use_cloudmask:
            cloud_mask = torch.from_numpy(cloud_mask)
        #最后，如果 self.opts.load_size - self.opts.crop_size 大于零，将对图像进行裁剪操作。裁剪的方式根据是否处于测试模式和随机性来决定，以获取指定大小的图像块。
        #表示加载图像时的大小减去裁剪图像时的大小，它计算了图像在加载后需要进行的裁剪操作的尺寸。
        if self.opts.load_size - self.opts.crop_size > 0:
            if not self.opts.is_test:
                y = random.randint(0, np.maximum(0, self.opts.load_size - self.opts.crop_size))
                x = random.randint(0, np.maximum(0, self.opts.load_size - self.opts.crop_size))
            else:
                #//2 表示将结果除以2，即取它的一半，这将使裁剪位置位于图像垂直中心。
                y = np.maximum(0, self.opts.load_size - self.opts.crop_size)//2
                x = np.maximum(0, self.opts.load_size - self.opts.crop_size)//2
            s1_data = s1_data[...,y:y+self.opts.crop_size,x:x+self.opts.crop_size]
            #示在可能存在多个维度的情况下，保持其他维度不变，只对垂直和水平方向进行裁剪。
            s2_cloudfree_data = s2_cloudfree_data[...,y:y+self.opts.crop_size,x:x+self.opts.crop_size]
            s2_cloudy_data = s2_cloudy_data[...,y:y+self.opts.crop_size,x:x+self.opts.crop_size]
            if self.opts.is_use_cloudmask:
                cloud_mask = cloud_mask[y:y+self.opts.crop_size,x:x+self.opts.crop_size]
                #将处理后的图像数据和其他信息（如文件名）存储在 results 字典中，并返回该字典作为函数的输出。
                # 这个 results 字典包含了云层图像、无云层图像、SAR 图像以及可能的云掩膜等信息，以便后续的模型训练或测试使用。
        results = {'cloudy_data': s2_cloudy_data,
                   'cloudfree_data': s2_cloudfree_data,
                   'SAR_data': s1_data,
                   'file_name': fileID[4]}
        if self.opts.is_use_cloudmask:
            results['cloud_mask'] = cloud_mask
        return results

    #返回数据集的长度，即包含的数据项数量。
    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images
    #用于加载光学图像数据，通过调用Rasterio库打开图像文件。
    def get_opt_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np .isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image

    #用于加载SAR图像数据，同样通过Rasterio库打开图像文件。
    def get_sar_image(self, path):
        src = rasterio.open(path, 'r', driver='GTiff')
        #于读取打开的图像文件，返回图像数据。
        image = src.read()
        #用于关闭图像文件。
        src.close()
        #用于填充图像中的NaN（非数字）值，将它们替换为图像的平均值。这一步骤有助于处理图像中的缺失数据或异常值。
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image

    #用于对数据进行标准化。
    # 对不同类型的数据（SAR、光学）进行不同的处理，包括裁剪、归一化等。
    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            #对数据中的每个通道进行处理，通道的数量由 len(data_image) 给出。
            for channel in range(len(data_image)):
                #它使用 np.clip 函数将每个通道的值限制在min与max的范围之内，这个范围限制是为了确保数据的值不会超出指定的最小和最大值。
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
                #它将每个通道的值减去 self.clip_min[data_type - 1][channel]，确保数据的值不会超出指定的最小和最大值
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                #将每个通道的值缩放到0到self.max_val之间，以进行归一化。
                data_image[channel] = self.max_val * (data_image[channel] / (self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
                # 将剪切后的像素值除以self.scale，从而将像素值缩放到[0, 1]的范围内，进行归一化。
            data_image /= self.scale

        return data_image
'''
read data.csv
'''

#用于读取数据文件列表total.csv中的训练集、验证集和测试集。
def get_train_val_test_filelists(listpath):

    csv_file = open(listpath, "r")
    #创建一个CSV读取器，用于逐行读取CSV文件中的内容。
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        #获取当前行的所有条目。
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)

    csv_file.close()

    return train_filelist, val_filelist, test_filelist

if __name__ == "__main__":
    ##===================================================##
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    # parser.add_argument('--input_data_folder', type=str, default='/home2/lsy/dataset/cloud removal')
    # parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/dataset/cloud removal/total.csv')
    parser.add_argument('--input_data_folder', type=str, default='/data/nyc/SEN12MS-CR/Dataset')
    parser.add_argument('--data_list_filepath', type=str, default='/data/nyc/SEN12MS-CR/Dataset/total.csv')
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=True)
    #云阈值，用于云掩膜的阈值设定，默认为0.2。
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    opts = parser.parse_args() 

    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ##===================================================##
    train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    ##===================================================##
    #传递解析后的命令行参数opts和test_filelist作为数据集的输入
    data = AlignedDataset(opts, test_filelist)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=4,shuffle=False)

    ##===================================================##
    _iter = 0
    for results in dataloader:
        cloudy_data = results['cloudy_data']
        cloudfree_data = results['cloudfree_data']
        SAR = results['SAR_data']
        if opts.is_use_cloudmask:
            cloud_mask = results['cloud_mask']
        file_name = results['file_name']
        print(_iter, file_name)
        print('cloudy:', cloudy_data.shape)
        print('cloudfree:', cloudfree_data.shape)
        print('sar:', SAR.shape)
        if opts.is_use_cloudmask:
            print('cloud_mask:', cloud_mask.shape)
        _iter += 1
