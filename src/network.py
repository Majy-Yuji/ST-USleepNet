import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import GraphUnet, Initializer, norm_g

class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, config):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, config.act_n)()
        self.c_act = getattr(nn, config.act_c)()
        self.g_unet = GraphUnet(
            config.ks, config.cs, config.kernal, config.chs, in_dim, self.n_act, config.drop_n)
        self.out_l = nn.Linear(in_dim, n_classes)
        self.out_drop = nn.Dropout(p = config.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels):
        # gs 是图数据，hs 是节点数据
        hs = self.embed(gs, hs)
        logits = self.classify(hs)
        return self.metric(logits, labels)

    def embed(self, gs, hs):
        gs = torch.stack(gs, dim=0)
        hs = torch.stack(hs, dim=0)
        gs = norm_g(gs)
        hs = self.g_unet(gs, hs)
        return hs

    # def embed(self, gs, hs):
    #     o_hs = []
    #     # gs 和 hs 是一个 batch 里的数据，分别拿出来处理
    #     for g, h in zip(gs, hs):
    #         h = self.embed_one(g, h)
    #         o_hs.append(h)
    #     hs = torch.stack(o_hs)
    #     return hs
    #
    # def embed_one(self, g, h):
    #     # 对图数据进行归一化
    #     g = norm_g(g)
    #     # 在这里进入 U-net
    #     hs = self.g_unet(g, h)
    #     return hs

    def classify(self, h):
        h = self.out_drop(h)
        # size: [batch_size, node_dim, feature_dim]
        # h, _ = h.max(dim = 1, keepdim = True)
        h = h.squeeze(dim = 1)
        h = h.mean(dim = 1, keepdim = True)
        # size: [batch_size, 1, feature_dim]
        h = h.squeeze(1)
        h = torch.relu(h)
        # size: [batch_size, feature_dim]
        h = self.out_l(h)
        # h = torch.relu(h)
        # size: [batch_size, n_classes]
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        class_acc = torch.zeros(5)  # 初始化每个类别的准确率为0
        class_num = torch.zeros(5)  # 初始化每个类别的准确率为0
        # 计算每个类别的 acc
        for i in range(5):  # 遍历每个类别
            correct = torch.sum((preds == i) & (labels == i)).float()
            total = torch.sum(labels == i).float()
            class_num[i] = total
            class_acc[i] = correct / total if total != 0 else 0.0
        return loss, acc, preds, class_acc, class_num