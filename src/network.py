import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import GraphUnet, Initializer, norm_g

class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, config):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, config.act_n)()
        self.c_act = getattr(nn, config.act_c)()
        self.num_slice = config.num_slice
        self.slice_width = config.num_node
        self.g_unet = GraphUnet(
            config.ks, config.cs, config.kernal, config.chs,
            config.gcn_h, config.l_n, in_dim, self.n_act, config.drop_n)
        self.outl = nn.Linear(3000, 1)
        self.out_drop = nn.Dropout(p = config.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels):
        hs, salient_spacial, salient_node, ts_trasaction = self.embed(gs, hs)
        logits, embedding = self.classify(hs)
        return (self.metric(logits, labels), embedding,
                salient_spacial, salient_node, ts_trasaction)

    def embed(self, gs, hs):
        gs = torch.stack(gs, dim=0)
        hs = torch.stack(hs, dim=0)
        gs = norm_g(gs)
        hs, salient_spacial, salient_node, ts_trasaction = self.g_unet(gs, hs)
        return hs, salient_spacial, salient_node, ts_trasaction

    def classify(self, h):
        h = self.out_drop(h)
        slices = [h[:, :, i * self.slice_width : (i + 1) * self.slice_width]
                   for i in range(self.num_slice)]
        embedding = torch.cat(slices, dim = 3)
        h = embedding.mean(dim = 2, keepdim = False)
        h = torch.relu(h)
        h = self.outl(h).squeeze()
        return F.log_softmax(h, dim = 1), embedding

    def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        class_acc = torch.zeros(5)
        class_num = torch.zeros(5)
        for i in range(5):
            correct = torch.sum((preds == i) & (labels == i)).float()
            total = torch.sum(labels == i).float()
            class_num[i] = total
            class_acc[i] = correct / total if total != 0 else 0.0
        return loss, acc, preds, class_acc, class_num