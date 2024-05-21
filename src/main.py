import argparse
import random
import time
import torch
import wandb
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('--cuda', default = 1, type = int, help='CUDA device number')
    parser.add_argument('--seed', type = int, default = 1, help = 'seed')
    parser.add_argument('--data', default = 'ISRUC_S3_10', help = 'data folder name')
    parser.add_argument('--fold', type = int, default = 2, help = 'fold (1..10)')
    parser.add_argument('--num_epochs', type = int, default = 100, help = 'epochs')
    parser.add_argument('--batch', type = int, default = 12, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 5e-4, help = 'learning rate')
    parser.add_argument('--drop_n', type = float, default = 0.3, help = 'drop net')
    parser.add_argument('--drop_c', type = float, default = 0.2, help = 'drop output')
    parser.add_argument('--act_n', type = str, default = 'ELU', help = 'network act')
    parser.add_argument('--act_c', type = str, default = 'ELU', help = 'output act')
    parser.add_argument('--ks', nargs = '+', default = '0.9 0.8 0.7')
    parser.add_argument('--cs', nargs = '+', default = '0.5 0.5 0.5')
    parser.add_argument('--sch', type=int, default = 2, help = 'scheduler')
    parser.add_argument('--chs', nargs = '+', default = '16 32 64 128')
    parser.add_argument('--kernal', type = int, default = 3, help = 'kernal')
    parser.add_argument('--wdb', type = bool, default = True)
    parser.add_argument('--sweep', type = bool, default = True)
    parser.add_argument('--weightDecay', type = float, default = 0.0008)
    parser.add_argument('--lrStepSize', type = int, default = 10)
    parser.add_argument('--lrGamma', type = int, default = 0.1)
    parser.add_argument('--lrFactor', type = float, default = 0.5)
    parser.add_argument('--lrPatience', type = int, default = 5)
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def app_run(args, config, G_data, fold_idx):
    # 通过添加 self.train 和 self.test 来划分训练集和测试集
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, config)
    trainer = Trainer(args, net, G_data)
    trainer.train()

def main():
    args = get_args()
    print(args)
    if args.sweep:
        wandb.init(
            project="Sleep Unet"
        )
        config = wandb.config
    else:
        wandb.init(
            project="Sleep Unet",
            config = args
        )
        config = args
    set_random(config.seed)
    start = time.time()
    G_data = FileLoader(config).load_data()
    print('load data using ------>', time.time()-start)
    if config.fold == 0:
        # 直接跑 10 折
        for fold_idx in range(10):
            print('start training ------> fold', fold_idx + 1)
            app_run(args, config, G_data, fold_idx)
    else:
        print('start training ------> fold', config.fold)
        app_run(args, config, G_data, config.fold-1)


if __name__ == "__main__":
    main()