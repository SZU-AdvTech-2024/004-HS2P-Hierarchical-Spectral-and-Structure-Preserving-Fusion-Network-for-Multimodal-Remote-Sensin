import numpy as np
import scipy
import scipy.signal as scisig


# naming conventions:
# ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
# ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2'])
# [        1,    2,      3,       4,     5,    6,    7,    8,     9,      10,            11,      12,     13]) #gdal
# [        0,    1,      2,       3,     4,    5,    6,    7,     8,      9,            10,      11,     12]) #numpy
# [              BB      BG       BR                       BNIR                                  BSWIR1    BSWIR2

# ge. Bands 1, 2, 3, 8, 11, and 12 were utilized as BB , BG , BR , BNIR , BSWIR1 , and BSWIR2, respectively.

#函数的目标是将输入数据 data 进行重新缩放，使其范围在0到1之间。
def get_rescaled_data(data, limits):
    #输入参数 data 是一个数据数组，可以是一维或多维的。
    #limits 是一个包含两个值的数组，表示数据的上限和下限。
    #这将数据从给定的范围缩放到0到1的范围。
    return (data - limits[0]) / (limits[1] - limits[0])


#计算两个通道之间的归一化差异。
def get_normalized_difference(channel1, channel2):
    #其中分子是两个通道的差值，分母是两个通道的和。
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    # get data between 0 and 1
    #这一行将输入的 data_image 数组中的所有值除以 10000，目的是将数据归一化到0到1的范围内。
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    #这一行创建一个与输入图像相同大小的零填充数组，并将其数据类型设置为 32 位浮点数。这个数组将用于存储阴影掩膜。
    shadow_mask = np.zeros((r, c)).astype('float32')

    #这几行分别从 data_image 中提取出三个不同通道的数据，并分别赋值给 BB、BNIR 和 BSWIR1 变量。这些通道数据将用于计算阴影。
    BB = data_image[1]
    BNIR = data_image[7]
    BSWIR1 = data_image[11]

    #：这一行计算出 BNIR 和 BSWIR1 通道的平均值，然后将结果赋给 CSI 变量。
    CSI = (BNIR + BSWIR1) / 2.

    #这两行计算了一个阈值 T3，它是基于 CSI 通道的最小值和平均值的线性组合。
    t3 = 3 / 4
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    #两行计算了另一个阈值 T4，它是基于 BB 通道的最小值和平均值的线性组合。
    t4 = 5 / 6
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    #这一行创建了一个布尔值数组 shadow_tf，用于指示哪些像素被认为是阴影。它基于 CSI 和 BB 通道的值与 T3 和 T4 的比较结果。
    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    #这一行将 shadow_mask 数组中 shadow_tf 为 True 的像素位置标记为 -1，用来表示这些像素是阴影。
    shadow_mask[shadow_tf] = -1
    #最后，这一行使用中值滤波器对 shadow_mask 进行平滑处理，以进一步处理和增强阴影掩膜。
    shadow_mask = scisig.medfilt2d(shadow_mask, 5)

    return shadow_mask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    #创建一个与输入图像相同大小的全1数组，并将其数据类型设置为 32 位浮点数。这个数组将用于存储云的可能性分数。
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    #以下几行计算不同通道的亮度分数，并将其存储在 score 数组中。这些分数用于评估每个像素是否属于云：
    score = np.minimum(score, get_rescaled_data(data_image[1], [0.1, 0.5]))#计算蓝色通道的亮度分数。
    score = np.minimum(score, get_rescaled_data(data_image[0], [0.1, 0.3]))#计算绿色通道的亮度分数。
    score = np.minimum(score, get_rescaled_data((data_image[0] + data_image[10]), [0.15, 0.2]))#计算组合通道的亮度分数。
    # Clouds are reasonably bright in all visible bands.
    score = np.minimum(score, get_rescaled_data((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))#计算可见光通道的亮度分数。

    #如果 use_moist_check 为真，将进行额外的湿度检查。这部分代码计算了湿度相关的分数并更新 score 数组。
    if use_moist_check:
        # Clouds are moist
        #这行代码计算了归一化差异指数（NDMI），它用于检测图像中的湿度。NDMI是通过从通道7和通道11的数据图像中获取像素值，
        ndmi = get_normalized_difference(data_image[7], data_image[11])
        #:这行代码将计算得到的 NDMI 分数与先前的 score 分数进行比较，并将较小的分数保留在 score 中。这一步是为了确保云中的湿度不会过高，以帮助进一步确定云区域。
        score = np.minimum(score, get_rescaled_data(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    # 这行代码计算了归一化雪指数（NDSI），用于检测云是否可能是雪。DSI 是通过从通道2和通道11的数据图像中获取像素值，
    ndsi = get_normalized_difference(data_image[2], data_image[11])
    #似于前一步，这行代码将计算得到的 NDSI 分数与先前的 score 分数进行比较，并将较小的分数保留在 score 中。这有助于排除雪覆盖区域被错误地标记为云。
    score = np.minimum(score, get_rescaled_data(ndsi, [0.8, 0.6]))

    #这两行代码定义了一个7x7大小的平均滤波器，用于平滑 score 图像。
    box_size = 7
    box = np.ones((box_size, box_size)) / (box_size ** 2)

    #这行代码执行灰度闭运算，通过将图像中的云区域连接起来并平滑边缘，以进一步改进云的检测结果。
    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    #这行代码使用平均滤波器 box 对 score 图像进行卷积操作，以进一步平滑图像并减少噪声。
    score = scisig.convolve2d(score, box, mode='same')

    #将 score 图像的像素值限制在0.00001和1.0之间，确保分数在有效范围内。
    score = np.clip(score, 0.00001, 1.0)

    #如果 binarize 参数为真，这段代码将执行二进制化操作。
    # 所有得分高于等于 cloud_threshold 的像素将被标记为1，而得分低于 cloud_threshold 的像素将被标记为0。
    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold] = 0
    #最终，函数返回一个表示云掩膜或云概率分数的数组，
    # 具体操作取决于是否进行了二进制化（binarize 参数）和湿度检查（use_moist_check 参数）。
    # 此掩膜可用于确定图像中的云区域。
    return score


def get_cloud_cloudshadow_mask(data_image, cloud_threshold):
    #并将结果进行二进制化（binarize=True），将像素值大于等于阈值的部分标记为1，表示云，小于阈值的部分标记为0。
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True)
    #取云阴影掩膜。它使用输入的 data_image 数据，并返回一个表示云阴影的掩膜。
    shadow_mask = get_shadow_mask(data_image)

    # 这行代码创建一个与云掩膜 cloud_mask 相同大小的全零数组，用于存储最终的云和云阴影掩膜。
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    #这行代码将云阴影掩膜中的像素值小于0（通常表示阴影）的部分标记为-1，表示云阴影。
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    #: 这行代码将云掩膜中的像素值大于0（通常表示云）的部分标记为1，表示云。
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    return cloud_cloudshadow_mask