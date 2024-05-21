import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from utils.dataset import GraphData
from collections import Counter

class Trainer:
    def __init__(self, args, net, G_data):
        if args.sweep:
            self.config = wandb.config
        else:
            self.config = args
            if args.wdb:
                config = {
                    "cuda": args.cuda,
                    "seed": args.seed,
                    "data": args.data,
                    "fold": args.fold,
                    "num_epochs": args.num_epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "drop_n": args.drop_n,
                    "drop_c": args.drop_c,
                    "act_n": args.act_n,
                    "ks": args.ks,
                    "cs": args.cs,
                    "chs": args.chs,
                    "sch": args.sch,
                    "kernal": args.kernal,
                    "weightDecay": args.weightDecay,
                    "lrStepSize": args.lrStepSize,
                    "lrGamma": args.lrGamma,
                    "lrFactor": args.lrFactor,
                    "lrPatience": args.lrPatience
                }
                wandb.config.update(config)
        self.cuda = self.config.cuda
        self.net = net
        self.feat_dim = G_data.feat_dim
        self.fold_idx = G_data.fold_idx
        self.init(G_data.train_gs, G_data.test_gs)
        self.device = self.set_device()
        self.net.to(self.device)
        self.wdb = self.config.wdb
        self.sch = self.config.sch
        self.log_file = 'logs//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt'

    def init(self, train_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.config.batch, True)
        self.test_d = test_data.loader(self.config.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr = self.config.lr, amsgrad = True,
            weight_decay = self.config.weightDecay)
        if self.config.sch == 1:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size = self.config.lrStepSize,
                                                       gamma = self.config.lrGamma)
        elif self.config.sch == 2:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode = 'min',
                                                                  factor = self.config.lrFactor,
                                                                  patience = self.config.lrPatience,
                                                                  verbose = True)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.to(self.device) for g in gs]
            return gs.to(self.device)
        return gs

    def set_device(self):
        if torch.cuda.is_available() and self.cuda is not None:
            torch.cuda.set_device(self.cuda)
            device = torch.device("cuda")
            print(
                f"Using CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
        return device

    def run_epoch(self, epoch, data, model, optimizer, metric_dict):
        # with open('Other//train_eval//' + str(epoch) + '_train.txt', 'w') as f:
        #     for name, param in model.named_parameters():
        #         f.write(f"Layer: {name} | Size: {param.size()} | Values : {param[:2].tolist()} \n")
        losses, accs, n_samples = [], [], 0
        stage_samples = torch.zeros(5)
        class_accs = torch.zeros(5)
        # labels = []
        # preds = []
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys = batch
            # labels.append(ys)
            # cur_len 是当前 batch 的长度
            # gs 是 g.A 的 list
            # hs 是 g.fea 的 list
            # ys 是 label 的 list
            # 把数据放到服务器上
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            loss, acc, pred, class_acc, class_num = model(gs, hs, ys)
            # preds.append(pred)
            losses.append(loss * cur_len)
            accs.append(acc * cur_len)
            stage_samples = stage_samples + class_num
            class_accs = class_accs + class_acc * class_num
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        class_acc = class_accs / stage_samples
        if optimizer is not None:
            metric_dict['train_acc'] = avg_acc.item()
            metric_dict['train_loss'] = avg_loss.item()
            metric_dict['train_acc_Weak'] = class_acc[0]
            metric_dict['train_acc_N1'] = class_acc[1]
            metric_dict['train_acc_N2'] = class_acc[2]
            metric_dict['train_acc_N3'] = class_acc[3]
            metric_dict['train_acc_Rem'] = class_acc[4]
        else:
            metric_dict['test_acc'] = avg_acc.item()
            metric_dict['test_loss'] = avg_loss.item()
            metric_dict['test_acc_Weak'] = class_acc[0]
            metric_dict['test_acc_N1'] = class_acc[1]
            metric_dict['test_acc_N2'] = class_acc[2]
            metric_dict['test_acc_N3'] = class_acc[3]
            metric_dict['test_acc_Rem'] = class_acc[4]
        # concatenated_label = torch.cat(labels, dim=0)
        # one_dimensional_label = concatenated_label.view(-1)
        # # print(Counter(yss_list))
        # label_list = one_dimensional_label.tolist()
        # concatenated_pred = torch.cat(preds, dim=0)
        # one_dimensional_pred = concatenated_pred.view(-1)
        # pred_list = one_dimensional_pred.tolist()
        # df = pd.DataFrame({'pred': pred_list, 'label': label_list})
        # if optimizer is not None:
        #     df.to_csv('Other//train_eval//' + str(epoch) + '_train.csv')
        # else:
        #     df.to_csv('Other//train_eval//' + str(epoch) + '_eval.csv')
        return avg_loss.item(), avg_acc.item(), metric_dict, class_acc, stage_samples

    def train(self):
        max_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'
        line_str = '%d:\t%.5f\n'
        stage_acc_str = 'Weak acc %.5f N1 acc %.5f N2 acc %.5f N3 acc %.5f REM acc %.5f'
        stage_num_str = 'Weak num %d N1 num %d N2 num %d N3 num %d REM num %d'
        for e_id in range(self.config.num_epochs):
            metric_dict = {}
            self.net.train()
            loss, acc, metric_dict, class_acc, stage_samples = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer, metric_dict)
            print(train_str % (e_id, loss, acc))
            print(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
            print(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                   stage_samples[3], stage_samples[4]))
            if self.wdb:
                if e_id == 0:
                    train_label = {"train_Weak": stage_samples[0],
                                   "train_N1": stage_samples[1],
                                   "train_N2": stage_samples[2],
                                   "train_N3": stage_samples[3],
                                   "train_REM": stage_samples[4]}
                    wandb.config.update(train_label)

            with open(self.log_file, 'a+') as f:
                f.write(train_str % (e_id, loss, acc))
                f.write(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
                f.write(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                         stage_samples[3], stage_samples[4]))

            self.net.eval()
            with torch.no_grad():
                loss, acc, metric_dict, class_acc, stage_samples = self.run_epoch(
                    e_id, self.test_d, self.net, None, metric_dict)
                if self.config.sch == 1:
                    self.scheduler.step()
                elif self.config.sch == 2:
                    self.scheduler.step(loss)
            max_acc = max(max_acc, acc)
            print(test_str % (e_id, loss, acc, max_acc))
            print(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
            print(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                   stage_samples[3], stage_samples[4]))
            if self.wdb:
                wandb.log(metric_dict)
                if e_id == 0:
                    test_label = {"test_Weak": stage_samples[0],
                                  "test_N1": stage_samples[1],
                                  "test_N2": stage_samples[2],
                                  "test_N3": stage_samples[3],
                                  "test_REM": stage_samples[4]}
                    wandb.config.update(test_label)

            with open(self.log_file, 'a+') as f:
                f.write(test_str % (e_id, loss, acc, max_acc))
                f.write(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
                f.write(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                         stage_samples[3], stage_samples[4]))

        with open(self.log_file, 'a+') as f:
            f.write(line_str % (self.fold_idx, max_acc))
