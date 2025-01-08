import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Training HS2P model.")

# 添加参数
# parser.add_argument('--epoch', type=int, default=61, help='batch size used for training')
parser.add_argument('--epoch', type=int, default=31, help='batch size used for training')

parser.add_argument('--start_epoch', type=int, default=1, help='batch size used for start  training')
parser.add_argument('--load_size', type=int, default=256)
# parser.add_argument('--load_size', type=int, default=512)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--num_workers', type = int, default=0, help='num_worker')

parser.add_argument('--lr', type=float, default=0.00007)
# parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
# parser.add_argument('--input_data_folder', type=str, default='/home2/lsy/datasets/SEN12MS-CR')
# parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/SEN12MS-CR/output/split_44.csv')
# parser.add_argument('--input_data_folder', type=str, default='/home2/lsy/datasets/SEN12MS-CR')
# parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/SEN12MS-CR/total.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/SEN12MS-CR/zrh/cloud_30_40.csv')
# parser.add_argument('--input_data_folder', type=str, default='/home7/lsy/datasets/cloud removal dataset')
# parser.add_argument('--data_list_filepath', type=str, default='/home7/lsy/datasets/cloud removal dataset/total.csv')
# parser.add_argument('--input_data_folder', type=str, default='/home7/lsy/datasets/cloud removal dataset')
# parser.add_argument('--data_list_filepath', type=str, default='/home7/lsy/datasets/cloud removal dataset/1w_datasets_new.csv')
# parser.add_argument('--input_data_folder', type=str, default='/home2/lsy/datasets/SEN12MS-CR')
# parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/SEN12MS-CR/1w_datasets_new.csv')


# parser.add_argument('--input_data_folder', type=str, default='/home2/lsy/datasets/new')
# parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/sentinel.csv')
parser.add_argument('--input_data_folder', type=str, default='/data/nyc/SEN12MS-CR/Dataset')
parser.add_argument('--data_list_filepath', type=str, default='/data/nyc/SEN12MS-CR/Dataset/total.csv')

#parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/figure.csv')
# parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/1_9_50_45_17.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/new_total_prob.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/probability.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/sentinel2.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home2/lsy/datasets/new/figure.csv')
# parser.add_argument('--input_data_folder', type=str, default='/home5/lsy/datasets/new')
# parser.add_argument('--data_list_filepath', type=str, default='/home5/lsy/datasets/new/sentinel.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home5/lsy/datasets/new/figure.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home7/lsy/pycodes/HS2P/cloud_ratio/cloud_0_20.csv')
#parser.add_argument('--data_list_filepath', type=str, default='/home7/lsy/pycodes/HS2P/cloud_ratio/cloud_40_50.csv')
#parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
parser.add_argument('--is_use_cloudmask', type=bool, default=True)
parser.add_argument('--cloud_threshold', type=float, default=0.2)

parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')
parser.add_argument('--warmup', action='store_true', default=True, help='warmup')

