import os
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import *
# from utils.dataset_utils import MixUp_AUG
from utils.indicator import *
from utils.loss import hs2p_loss
# from utils.loss_zrh import spec_spac_loss
from torch.utils.tensorboard import SummaryWriter
# from models.net import pixel_reshuffle
import pandas as pd

from warmup_scheduler import GradualWarmupScheduler


class Trainer:
    def __init__(self, name, log_dir='logs', csv_file='avg_maxg_midg_in_series_256_logs.csv'):
        self.name = name
        self.csv_file = csv_file
        print("Start training " + str(name))
        self.writer = SummaryWriter(log_dir=log_dir)  # 初始化 TensorBoard SummaryWriter

        # 初始化CSV文件，写入标题
        # df = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Val Loss', 'PSNR', 'SSIM'])
        # df.to_csv(self.csv_file, index=False)
        # 初始化CSV文件，只有在文件不存在时才写入标题
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Val Loss', 'PSNR', 'SSIM'])
            df.to_csv(self.csv_file, index=False)

    def adjust_state_dict(self, state_dict, net):
        """调整状态字典的键以适应模型的期望."""
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        # 检查模型是否被 DataParallel 包装
        model_is_data_parallel = hasattr(net, 'module')

        for key, value in state_dict.items():
            # 移除或添加 'module.' 前缀
            if model_is_data_parallel and not key.startswith('module.'):
                new_key = 'module.' + key
            elif not model_is_data_parallel and key.startswith('module.'):
                new_key = key.replace('module.', '')
            else:
                new_key = key
            new_state_dict[new_key] = value

        return new_state_dict

    def train(self,device,net,args,optimizer,scheduler):

        train_filelist, val_filelist, _ = get_train_val_test_filelists(args.data_list_filepath)
        train_data = AlignedDataset(args, train_filelist)#训练集长度101615
        val_data = AlignedDataset(args, val_filelist)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_sz,shuffle=True, num_workers=args.num_workers)#101615/batchsize
        val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_sz, shuffle=False, num_workers=args.num_workers)

        print(len(train_dataloader) * args.batch_sz)
        print(len(val_dataloader) * args.batch_sz)
        print("数据加载完成,开始训练")


        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Adjust the state dictionary to match the model's expected keys
            adjusted_state_dict = self.adjust_state_dict(checkpoint['model_state_dict'], net)

            # Load the adjusted state dictionary
            net.load_state_dict(adjusted_state_dict)

            # 恢复学习率
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = checkpoint.get('lr', param_group['lr'])
                print(f"Restored learning rate for param group {i}: {param_group['lr']}")

            # Load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 48, eta_min=1e-6,
                                                                     last_epoch=12)

            #scheduler.load_state_dict(checkpoint['scheduler_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
        else:
            args.start_epoch = 1


        train_step = 0
        val_step = 0
        for epoch in range(args.start_epoch, args.epoch):
            net.train()
            training_loss = 0
            num_batch = 0
            train_psnr = 0

            # 训练一个epoch
            for i, triple_pack in enumerate(train_dataloader):#列表5 cloudy_data:24 13 128 128 cloudfree_data:24 13 128 128 sar_data:24 13 128 128 filename:24个文件名 cloud_mask 24 128 128
                cloudy_img = triple_pack["cloudy_data"].to(device)
                cloudfree_img = triple_pack["cloudfree_data"].to(device)
                sar_img = triple_pack["SAR_data"].to(device)
                cloud_mask = triple_pack["cloud_mask"].to(device)
                cloud_mask = torch.unsqueeze(cloud_mask, dim=1)

                optimizer.zero_grad()
                predict_img = net(sar_img,cloudy_img).to(device)
                batch_loss = hs2p_loss(cloudfree_img, predict_img).to(device)


                training_loss = training_loss + batch_loss.item()
                num_batch = num_batch + 1
                train_step += 1

                batch_loss.backward()
                optimizer.step()

                if train_step % 100 == 0:
                    print(f"step: {train_step}, batch_loss: {batch_loss.item()}")

            scheduler.step()#自动更新当前 epoch 的学习率
            #scheduler.step(epoch)#手动指定 epoch 来更新学习率
            torch.cuda.empty_cache()  # 释放显存

            training_loss /= num_batch

            # print('Epoch [{}/{}], Training Loss: {:.8f}, lr: {:.12f}'
            #       .format(epoch, args.epoch, training_loss, scheduler.get_last_lr()[0]))
            print('Epoch [{}/{}], Training Loss: {:.8f}, lr: {:.12f}'
                  .format(epoch, args.epoch, training_loss, optimizer.param_groups[0]['lr']))


            if epoch % 1 == 0 and epoch != 0:
                # 在验证前分配显存占位，确保每张 GPU 占用大约 25GB 的显存
                net.eval()
                psnr = 0
                ssim = 0
                val_loss = 0
                val_num_batch = 0

                with torch.no_grad():
                    for i, triple_pack in enumerate(val_dataloader):
                        cloudy_img = triple_pack["cloudy_data"].to(device)
                        cloudfree_img = triple_pack["cloudfree_data"].to(device)
                        sar_img = triple_pack["SAR_data"].to(device)

                        predict = net(sar_img, cloudy_img).to(device)
                        val_batch_loss = hs2p_loss(cloudfree_img, predict).to(device)
                        val_loss += val_batch_loss.item()
                        val_num_batch += 1
                        val_step += 1

                        predict = predict[:,3*13:4*13, :, :]
                        psnr = psnr + PSNR(predict, cloudfree_img)
                        ssim = ssim + SSIM(predict, cloudfree_img,device)

                    val_loss /= val_num_batch
                    psnr /= len(val_dataloader)
                    ssim /= len(val_dataloader)

                    print(f"PSNR: {psnr}")
                    print(f"SSIM: {ssim}")
                    print(f"Validation Loss: {val_loss}")
                    # 记录到 CSV 文件
                    with open(self.csv_file, 'a') as f:
                        df = pd.DataFrame([[epoch, training_loss, val_loss, psnr, ssim]],
                                          columns=['Epoch', 'Training Loss', 'Val Loss', 'PSNR', 'SSIM'])
                        df.to_csv(f, header=False, index=False)
                # 验证完成后释放占位显存
                    # 将 val_loss 写入 TensorBoard, 添加模型名字
                #self.writer.add_scalar(f'{self.name}/Loss/Train', epoch_loss, epoch)
                #将 train_loss 和 val_loss 写入 TensorBoard, 放在同一个图表中
                self.writer.add_scalars(f'{self.name}/Loss',
                                        {'Training': training_loss,
                                         'Validation': val_loss}, global_step=epoch)

                # # 保存模型
                # model_path = f'/home7/lsy/pycodes/HS2P/ckpts/abliation_experiment/avg_maxg_midg_in_series_256/model_{epoch}.ckpt'
                # 保存模型
                model_path = f'/home/nyc/hs2p1/ckpts/model_{epoch}.ckpt'
                # torch.save(net.state_dict(), model_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    #'lr': scheduler.get_last_lr()[0],  # 保存当前的学习率

                }, model_path)
                net.train()

            # 关闭 SummaryWriter
        self.writer.close()

