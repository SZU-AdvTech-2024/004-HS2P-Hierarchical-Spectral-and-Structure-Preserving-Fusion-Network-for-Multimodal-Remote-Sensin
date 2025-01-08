# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import cv2
# import random

from matplotlib import pyplot as plt
from torch import nn
import csv
# from models.baseline import baseline
from models.net0 import HS2P, get_gradient
# from models.net_concat import HS2P, get_gradient
from utils.arg_parser import parser
from utils.dataloader import *
from utils.metrics import *
import torch
from torch.cuda.amp import autocast

parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for testing')
# parser.add_argument('--model', default="/home2/lsy/pycodes/HS2P/ckpts/abliation_experiment/concat/model_44.ckpt", help='to predict')
parser.add_argument('--model', default="/home/nyc/hs2p1/ckpts/model_29.ckpt", help='to predict')
parser.add_argument('--is_test', type=bool, default=True)
args = parser.parse_args()


def remove_module_prefix(state_dict):
    """
    当模型训练时使用了nn.DataParallel，会在权重键前增加'module.'前缀。
    如果在不使用DataParallel的情况下加载权重，需要移除这个前缀。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]  # 去除前缀
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def generate_heatmap(attn, head=0, colormap=cv2.COLORMAP_JET, save_path=None, subset_size=512):
    """
    将注意力矩阵的中间部分裁剪为 256x256 并可视化为热力图。

    参数：
    attn: 注意力权重 (batch_size, heads, seq_len, seq_len) 的张量
    head: 要可视化的注意力头编号
    colormap: OpenCV 的 colormap 类型，如 COLORMAP_JET
    save_path: 如果提供保存路径，将热力图保存为图像文件
    subset_size: 子集的大小，默认取 256x256
    """
    # 假设我们只想可视化第一个 batch 的第一个注意力头
    attn_head = attn[0, head].detach().cpu().numpy()  # (seq_len, seq_len)

    # 确认 attn 矩阵大小为 16385x16385
    seq_len = attn_head.shape[0]
    assert seq_len == 16385, f"注意力矩阵的尺寸应为 16385x16385，但收到的是 {seq_len}x{seq_len}"

    # 计算中间部分的起始索引和结束索引
    center_start = (seq_len - subset_size) // 2
    center_end = center_start + subset_size

    # 裁剪出中间的 256x256 矩阵
    attn_subset = attn_head[center_start:center_end, center_start:center_end]

    # 将注意力矩阵缩放到 0-255 的范围
    attn_rescaled = (attn_subset - attn_subset.min()) / (attn_subset.max() - attn_subset.min())
    attn_rescaled = (attn_rescaled * 255).astype(np.uint8)

    # 应用 OpenCV colormap
    heatmap = cv2.applyColorMap(attn_rescaled, colormap)

    # 可选：展示热力图
    plt.imshow(heatmap)
    plt.axis('off')
    plt.show()

    # 如果指定了保存路径，将热力图保存到文件
    if save_path:
        cv2.imwrite(save_path, heatmap)


def predict(opts, net):
    # 创建一个保存提取通道图像的目录（如果不存在的话）
    output_dir = '/home/nyc/hs2p1/ckpts/myimagesrmse'
    # output_dir = '/home2/lsy/pycodes/HS2P/ckpts/images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)

    #多卡推理
    # checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage.cuda())
    # loaded_state_dict = checkpoint['model_state_dict']  # 获取模型权重
    # net.load_state_dict(loaded_state_dict)

    #载入训练好的权重   多卡推理的时候才需要去除'module.' 前缀
    checkpoint = torch.load(args.model, map_location=device)
    loaded_state_dict = checkpoint['model_state_dict']  # 获取实际的模型权重
    new_state_dict = remove_module_prefix(loaded_state_dict)  # 去除 'module.' 前缀

    #旧版载入权重
    # loaded_state_dict = torch.load(args.model, map_location=device)
    # new_state_dict = remove_module_prefix(loaded_state_dict)

    # baseline载入权重
    # loaded_state_dict = torch.load(args.model, map_location=device)
    # net.load_state_dict(loaded_state_dict)

    # 使用更新后的状态字典加载模型权重
    net.load_state_dict(new_state_dict)

    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    test_data = AlignedDataset(opts, test_filelist)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opts.batch_sz, shuffle=False, num_workers=args.num_workers )

    print(f"数据加载完成, 测试集长度为{len(test_dataloader) * opts.batch_sz}")
    # 初始化全局计数器，移到循环外部
    global_counter = 0

    psnr = 0
    ssim = 0
    mae = 0
    rmse = 0
    sam = 0
    lpips = 0
    cc = 0
    results_list = []  # 用于保存所有的结果



    with torch.no_grad():
        net.eval()
        # triple_pack:'cloudy_data'1 13 128 128,'cloudfree_data'1 13 128 128,'SAR_data'1 2 128 128,'file_name'1,'cloud_mask'1 128 128
        for i, triple_pack in enumerate(test_dataloader):
            cloudy_img = triple_pack["cloudy_data"].to(device)
            cloudfree_img = triple_pack["cloudfree_data"].to(device)
            sar_img = triple_pack["SAR_data"].to(device)
            fname = triple_pack["file_name"][0]
            # print(sar_img.device)
            # print(cloudy_img.device)

            predict_img = net(sar_img,cloudy_img).to(device)#1 104 128 128
            print(fname)


            #热力图输出
            # predict_img, r_attn, e_attn = net(sar_img,cloudy_img)
            # predict_img = predict_img.to(device)
            # r_attn = r_attn.to(device)
            # e_attn = e_attn.to(device)
            predict_img = predict_img[:, 3 * 13:4 * 13, :, :]#1 13 128 128
            # print(fname)
            #
            # # 路径
            # r_attn_save_dir = "/home7/lsy/pycodes/HS2P/ckpts/r_attn_images"
            # e_attn_save_dir = "/home7/lsy/pycodes/HS2P/ckpts/e_attn_images"
            #
            # # 确保路径存在，不存在则创建
            # os.makedirs(r_attn_save_dir, exist_ok=True)
            # os.makedirs(e_attn_save_dir, exist_ok=True)
            #
            #
            # # 生成保存路径，确保每次文件名唯一
            # r_attn_file_path = os.path.join(r_attn_save_dir, f"r_attn_heatmap_{global_counter}.png")
            # e_attn_file_path = os.path.join(e_attn_save_dir, f"e_attn_heatmap_{global_counter}.png")
            #
            # # 生成 r_attn 的热力图并保存
            # generate_heatmap(r_attn, head=0, colormap=cv2.COLORMAP_JET, save_path=r_attn_file_path, subset_size=512)
            #
            # # 生成 e_attn 的热力图并保存
            # generate_heatmap(e_attn, head=0, colormap=cv2.COLORMAP_JET, save_path=e_attn_file_path, subset_size=512)
            #
            # # 每次生成热力图后增加计数器，确保下次保存时文件名不同
            # global_counter += 1

            psnr_tmp = PSNR(cloudfree_img,predict_img)
            psnr = psnr + psnr_tmp
            print(f"PSNR: {psnr_tmp}")

            ssim_tmp = SSIM(cloudfree_img,predict_img,device).item()
            ssim = ssim + ssim_tmp
            print(f"SSIM: {ssim_tmp}")

            mae_tmp = cloud_mean_absolute_error(cloudfree_img,predict_img).item()
            mae = mae + mae_tmp
            print(f"MAE: {mae_tmp}")

            rmse_tmp = RMSE(cloudfree_img,predict_img)
            rmse = rmse + rmse_tmp
            print(f"RMSE: {rmse_tmp}")

            sam_tmp = cloud_mean_sam(cloudfree_img,predict_img).item()
            sam = sam + sam_tmp
            print(f"SAM: {sam_tmp * 180 / math.pi}")

            # 保存每个文件的结果
            results_list.append([psnr_tmp, ssim_tmp, sam_tmp * 180 / math.pi, mae_tmp, rmse_tmp, fname])

            # lpips_tmp = compute_LPIPS(cloudfree_img, predict_img,device)
            # lpips = lpips + lpips_tmp
            # print(f"LPIPS: {lpips_tmp}")
            #
            # cc_tmp = pcc(cloudfree_img, predict_img)
            # cc = cc + cc_tmp
            # print(f"cc: {cc}")

            #真实无云图的梯度图
            y_true_grad = get_gradient(cloudfree_img, 13)#1 13 128 128
            y_true_grad_data = y_true_grad.squeeze()[1:4, :, :] #3 128 128 # (3, h, w)
            if torch.min( y_true_grad_data) < 0:
                y_true_grad_data =  y_true_grad_data- torch.min( y_true_grad_data)

            y_true_grad_data =  y_true_grad_data / torch.max( y_true_grad_data) * 255

            y_true_grad_data =  y_true_grad_data.permute(1, 2, 0).detach().cpu().numpy()  # (h, w, c)

            # 使用线性变换增加亮度和对比度
            alpha = 1.5  # 对比度控制 (1.0-3.0，值越大对比度越大)
            beta = 30  # 亮度控制 (0-100，值越大亮度越大)

            # 调整对比度和亮度
            y_true_grad_data = cv2.convertScaleAbs(y_true_grad_data, alpha=alpha, beta=beta)

            # 准备保存预测结果
            # 对预测结果进行可视化处理
            pred_cloudfree_data = predict_img.squeeze()[1:4, :, :] # (3, h, w)#3 128 128#1 13 128 128->13 128 128->3 128 128 对前三个通道进行可视化
            if torch.min(pred_cloudfree_data) < 0:
                pred_cloudfree_data = pred_cloudfree_data - torch.min(pred_cloudfree_data)

            pred_cloudfree_data = pred_cloudfree_data / torch.max(pred_cloudfree_data) * 255

            pred_cloudfree_data = pred_cloudfree_data.permute(1, 2, 0).detach().cpu().numpy()  # (h, w, c)

            # 准备保存预测结果
            # 对有云图像进行可视化处理
            cloudy_data = cloudy_img.squeeze()[1:4, :, :]  # (3, h, w)
            if torch.min(cloudy_data) < 0:
                cloudy_data = cloudy_data - torch.min(cloudy_data)

            cloudy_data = cloudy_data / torch.max(cloudy_data) * 255

            cloudy_data = cloudy_data.permute(1, 2, 0).detach().cpu().numpy()  # (h, w, c)

            # 准备保存预测结果
            # 对gt图像像进行可视化处理
            cloudfree_data = cloudfree_img.squeeze()[1:4, :, :]  # (3, h, w)
            if torch.min(cloudfree_data) < 0:
                cloudfree_data = cloudfree_data - torch.min(cloudfree_data)

            cloudfree_data = cloudfree_data / torch.max(cloudfree_data) * 255

            cloudfree_data = cloudfree_data.permute(1, 2, 0).detach().cpu().numpy()  # (h, w, c)

            # 准备保存预测结果
            # 对有sar图像像进行可视化处理
            sar_data = sar_img.squeeze()[:2, :, :]  # 取前两个通道 (2, h, w)
            # 对两个通道取平均值，得到单通道图像 (h, w)
            sar_data = sar_data.mean(dim=0)  # 对第0维度（通道维度）进行求平均
            # 检查最小值并归一化到 0-255
            if torch.min(sar_data) < 0:
                sar_data = sar_data - torch.min(sar_data)

            sar_data = sar_data / torch.max(sar_data) * 255

            # 转换为 (h, w) 格式的 numpy 数组用于可视化
            sar_data = sar_data.detach().cpu().numpy()  # 转换为 numpy 格式


            # # 从原始文件名中去除 '.tif' 后缀
            if '.tif' in fname:
                fname = fname.replace('.tif', '')

            # 保存预测图像
            # pred_output_filename = f'pred_{fname}.png'
            # cloudy_output_filename = f'cloudy_{fname}.png'
            # cloudfree_output_filename = f'cloudfree_{fname}.png'
            pred_output_filename = f'{fname}_pred.png'
            cloudy_output_filename = f'{fname}_cloudy.png'
            cloudfree_output_filename = f'{fname}_cloudyfree.png'
            cloudfreegrad_output_filename = f'{fname}_cloudyfree_grad.png'
            sar_output_filename = f'{fname}_sar.png'

            pred_output_filepath = os.path.join(output_dir, pred_output_filename)
            cloudy_output_filepath = os.path.join(output_dir, cloudy_output_filename)
            cloudfree_output_filepath = os.path.join(output_dir, cloudfree_output_filename)
            cloudfreegrad_output_filepath = os.path.join(output_dir, cloudfreegrad_output_filename)
            sar_output_filepath = os.path.join(output_dir, sar_output_filename)

            cv2.imwrite(pred_output_filepath, pred_cloudfree_data)  # 保存图像
            cv2.imwrite(cloudy_output_filepath, cloudy_data)  # 保存图像
            cv2.imwrite(cloudfree_output_filepath, cloudfree_data)
            cv2.imwrite(cloudfreegrad_output_filepath, y_true_grad_data)
            cv2.imwrite(sar_output_filepath, sar_data)
            # print(f"Saved {output_filepath}")


    print('###############')
    print("##########################################")
    avg_psnr = psnr / len(test_dataloader)
    avg_ssim = ssim / len(test_dataloader)
    avg_mae = mae / len(test_dataloader)
    avg_rmse = rmse / len(test_dataloader)
    avg_sam = (sam / len(test_dataloader)) * 180 / math.pi
    # avg_lpips = (lpips) / len(test_dataloader)
    # avg_cc=(cc) / len(test_dataloader)


    print(f"AVG_PSNR: {avg_psnr}")
    print(f"AVG_SSIM: {avg_ssim}")
    print(f"AVG_MAE: {avg_mae}")
    print(f"AVG_RMSE: {avg_rmse}")
    print(f"AVG_SAM: {avg_sam}")
    # print(f"AVG_LPIPS: {avg_lpips }")
    # print(f"AVG_CC: {avg_cc}")
    print("##########################################")

    # 将结果按PSNR从高到低排序
    results_list.sort(key=lambda x: x[0], reverse=True)  # x[0] 为PSNR值
    # 将结果写入CSV文件
    csv_filename = os.path.join(output_dir, 'myresultsrmse.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PSNR', 'SSIM', 'SAM', 'MAE', 'RMSE', 'Filename'])
        for row in results_list:
            writer.writerow(row)
    print(f"Results saved to {csv_filename}")

    # 按PSNR从高到低排序并写入CSV文件
    # psnr_list.sort(key=lambda x: x[1], reverse=True)
    # with open(os.path.join(output_dir, 'psnr_results.csv'), mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Filename', 'PSNR'])
    #     for item in psnr_list:
    #         writer.writerow([item[0], item[1]])

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义网络
    net = HS2P(device, sar_channel=2, opt_channel=13, N=4)
    #net = baseline()
    net =net.to(device)
    #net = nn.DataParallel(net)
    predict(args,net)