import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from utils.dataset import GraphData
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

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
                    "gcn_h": args.gcn_h,
                    "l_n": args.l_n,
                    "delta_t": args.delta_t,
                    "delta_p": args.delta_p,
                    "num_class": args.num_class,
                    "feat_dim": args.feat_dim,
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
        self.init(G_data.train_gs, G_data.val_gs, G_data.test_gs)
        self.device = self.set_device()
        self.net.to(self.device)
        self.wdb = self.config.wdb
        self.sch = self.config.sch
        self.log_file = 'logs//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
        self.model_file = 'models//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.pth'

    def init(self, train_gs, val_gs, test_gs):
        print('#train: %d, #val: %d, #test: %d' % (len(train_gs), len(val_gs), len(test_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        val_data = GraphData(val_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.config.batch, True)
        self.val_d = val_data.loader(self.config.batch, False)
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

    def run_epoch(self, epoch, data, model, optimizer, metric_dict, if_explain, name):
        losses, accs, n_samples = [], [], 0
        stage_samples = torch.zeros(5)
        class_accs = torch.zeros(5)
        labels = []
        preds = []
        embeddings = []
        salient_spacials = []
        salient_nodes = []
        ts_trasactions = []
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys = batch
            labels.append(ys)
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            ([loss, acc, pred, class_acc, class_num],
             embedding, salient_spacial, salient_node, ts_trasaction) = model(gs, hs, ys)
            preds.append(pred)
            if if_explain:
                salient_spacials.append(salient_spacial)
                embeddings.append(embedding)
                salient_nodes.append(salient_node)
                ts_trasactions.append(ts_trasaction)
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
        concatenated_label = torch.cat(labels, dim=0)
        one_dimensional_label = concatenated_label.view(-1)
        label_list = one_dimensional_label.tolist()
        concatenated_pred = torch.cat(preds, dim=0)
        one_dimensional_pred = concatenated_pred.view(-1)
        pred_list = one_dimensional_pred.tolist()
        if if_explain:
            embedding = torch.cat(embeddings, dim = 0)
            embedding = F.softmax(embedding, dim = 1)
            class_indices = torch.argmax(embedding, dim = 1)
            salient_spacial = torch.cat(salient_spacials, dim = 0)
            salient_nodes = torch.cat(salient_nodes, dim = 0)
            ts_trasactions = torch.cat(ts_trasactions, dim = 0)
            return (avg_loss.item(), avg_acc.item(), metric_dict,
                    class_acc, stage_samples, label_list, pred_list,
                    class_indices, salient_spacial, salient_nodes, ts_trasactions)
        else:
            if name == 'train':
                metric_dict, f1, precision, recall, kappa = (
                    self.compute_metrics(label_list, pred_list, metric_dict, 'train'))
                metric_dict['train_acc'] = avg_acc.item()
                metric_dict['train_loss'] = avg_loss.item()
                metric_dict['train_acc_Weak'] = class_acc[0]
                metric_dict['train_acc_N1'] = class_acc[1]
                metric_dict['train_acc_N2'] = class_acc[2]
                metric_dict['train_acc_N3'] = class_acc[3]
                metric_dict['train_acc_Rem'] = class_acc[4]
            elif name == 'val':
                metric_dict, f1, precision, recall, kappa = (
                    self.compute_metrics(label_list, pred_list, metric_dict, 'val'))
                metric_dict['val_acc'] = avg_acc.item()
                metric_dict['val_loss'] = avg_loss.item()
                metric_dict['val_acc_Weak'] = class_acc[0]
                metric_dict['val_acc_N1'] = class_acc[1]
                metric_dict['val_acc_N2'] = class_acc[2]
                metric_dict['val_acc_N3'] = class_acc[3]
                metric_dict['val_acc_Rem'] = class_acc[4]
            else:
                metric_dict, f1, precision, recall, kappa = (
                    self.compute_metrics(label_list, pred_list, metric_dict, 'test'))
                metric_dict['test_acc'] = avg_acc.item()
                metric_dict['test_loss'] = avg_loss.item()
                metric_dict['test_acc_Weak'] = class_acc[0]
                metric_dict['test_acc_N1'] = class_acc[1]
                metric_dict['test_acc_N2'] = class_acc[2]
                metric_dict['test_acc_N3'] = class_acc[3]
                metric_dict['test_acc_Rem'] = class_acc[4]
            return (avg_loss.item(), avg_acc.item(), metric_dict, class_acc,
                    stage_samples, f1, precision, recall, kappa)

    def train(self):
        max_train_acc = 0.0
        max_val_acc = 0.0
        max_test_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f max %.5f\n'
        val_str = 'Val epoch %d: loss %.5f acc %.5f max %.5f\n'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f\n'
        stage_acc_str = 'Weak acc %.5f N1 acc %.5f N2 acc %.5f N3 acc %.5f REM acc %.5f\n'
        stage_num_str = 'Weak num %d N1 num %d N2 num %d N3 num %d REM num %d\n'
        metric_str = 'F1 %.5f Precision %.5f Recall %.5f Kappa %.5f\n'
        for e_id in range(self.config.num_epochs):
            metric_dict = {}
            self.net.train()
            loss, acc, metric_dict, class_acc, stage_samples, f1, precision, recall, kappa = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer, metric_dict, False, 'train')
            max_train_acc = max(max_train_acc, acc)
            metric_dict['max_train_acc'] = max_train_acc
            print(train_str % (e_id, loss, acc, max_train_acc))
            print(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
            print(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                   stage_samples[3], stage_samples[4]))
            print(metric_str % (f1, precision, recall, kappa))
            if self.wdb:
                if e_id == 0:
                    train_label = {"train_Weak": stage_samples[0],
                                   "train_N1": stage_samples[1],
                                   "train_N2": stage_samples[2],
                                   "train_N3": stage_samples[3],
                                   "train_REM": stage_samples[4]}
                    wandb.config.update(train_label)

            with open(self.log_file, 'a+') as f:
                f.write(train_str % (e_id, loss, acc, max_train_acc))
                f.write(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
                f.write(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                         stage_samples[3], stage_samples[4]))
                f.write(metric_str % (f1, precision, recall, kappa))
                f.write("\n")

            self.net.eval()
            with torch.no_grad():
                loss, acc, metric_dict, class_acc, stage_samples, f1, precision, recall, kappa = self.run_epoch(
                    e_id, self.val_d, self.net, None, metric_dict, False, 'val')
                if self.config.sch == 1:
                    self.scheduler.step()
                elif self.config.sch == 2:
                    self.scheduler.step(loss)
            max_val_acc = max(max_val_acc, acc)
            metric_dict['max_val_acc'] = max_val_acc
            if acc == max_val_acc:
                torch.save(self.net, self.model_file)
            print(val_str % (e_id, loss, acc, max_val_acc))
            print(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
            print(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                   stage_samples[3], stage_samples[4]))
            print(metric_str % (f1, precision, recall, kappa))
            if self.wdb:
                if e_id == 0:
                    val_label = {"val_Weak": stage_samples[0],
                                  "val_N1": stage_samples[1],
                                  "val_N2": stage_samples[2],
                                  "val_N3": stage_samples[3],
                                  "val_REM": stage_samples[4]}
                    wandb.config.update(val_label)

            with open(self.log_file, 'a+') as f:
                f.write(val_str % (e_id, loss, acc, max_val_acc))
                f.write(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
                f.write(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                         stage_samples[3], stage_samples[4]))
                f.write(metric_str % (f1, precision, recall, kappa))
                f.write("\n")

            # self.net.eval()
            with torch.no_grad():
                loss, acc, metric_dict, class_acc, stage_samples, f1, precision, recall, kappa = self.run_epoch(
                    e_id, self.test_d, self.net, None, metric_dict, False, 'test')
            max_test_acc = max(max_test_acc, acc)
            metric_dict['max_test_acc'] = max_test_acc
            print(test_str % (e_id, loss, acc, max_test_acc))
            print(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
            print(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                   stage_samples[3], stage_samples[4]))
            print(metric_str % (f1, precision, recall, kappa))

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
                f.write(test_str % (e_id, loss, acc, max_test_acc))
                f.write(stage_acc_str % (class_acc[0], class_acc[1], class_acc[2], class_acc[3], class_acc[4]))
                f.write(stage_num_str % (stage_samples[0], stage_samples[1], stage_samples[2],
                                         stage_samples[3], stage_samples[4]))
                f.write(metric_str % (f1, precision, recall, kappa))
                f.write("\n")

    def interpretability(self, model):
        loss, acc, metric_dict, class_acc, stage_samples,\
            label_list, pred_list, class_indices, salient_spacial_adjs,\
            salient_nodes, ts_trasactions = self.run_epoch(
            0, self.test_d, model, None, {}, True, 'test')
        return (acc, label_list, pred_list, class_indices, salient_spacial_adjs,
                salient_nodes, ts_trasactions)


    def compute_metrics(self, y_true, y_pred, metric_dict, name):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)

        metric_dict[name + '_F1_score'] = f1
        metric_dict[name + '_precision'] = precision
        metric_dict[name + '_recall'] = recall
        metric_dict[name + '_kappa'] = kappa
        return metric_dict, f1, precision, recall, kappa