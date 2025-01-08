import random

import math
import numpy as np
import torch
import os
from torch import nn,optim
# from thop import profile
# from thop import clever_format
# from torch.utils.tensorboard import  SummaryWriter
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
from models.net0 import HS2P
from trainer1 import Trainer
from utils.arg_parser import parser
# parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--batch_sz', type=int, default=24, help='batch size used for training')

#接着之前训练好的模型继续训练
parser.add_argument('--resume', default='/home/nyc/hs2p1/ckpts/model_4.ckpt', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default="/home/nyc/hs2p1/ckpts/model_4.ckpt", help='to predict')

# parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--is_test', type=bool, default=False)
args = parser.parse_args()

def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def count_params(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(f"device_count: {torch.cuda.device_count()}")


    set_seed_torch(42)
    net = HS2P(device, sar_channel=2, opt_channel=13, N=4)
    #net = baseline()
    net = net.to(device)
    # net = nn.DataParallel(net, device_ids=[2])

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()

    ######### Scheduler ###########
    if args.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = args.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch - warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    model = Trainer("avg_maxg_midg_in_series_256")

    #Print the total number of para and Macs
    # sar = torch.randn(1, 2, 128, 128).to(device)
    # cloudy=torch.randn(1, 13, 128, 128).to(device)
    # macs, params = profile(net.module, inputs=(sar,cloudy))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('Computational complexity:', macs)
    # print('Number of parameters: ', params)

    model.train(device,net,args,optimizer,scheduler)