import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def calculate_same_padding(kernel_size):
    return (kernel_size - 1) // 2

class GraphUnet(nn.Module):
    def __init__(self, ks, cs, kernal, chs, gcn_h, l_n, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = [float(num) for num in ks.split(" ")]
        self.cs = [float(num) for num in cs.split(" ")]
        self.gcn_h = [int(num) for num in gcn_h.split(" ")]
        self.channels = [1] + [int(num) for num in chs.split(" ")]
        self.kernal_size = [int(num) for num in kernal.split(" ")]
        self.down_cnns = nn.ModuleList()
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.gpools = nn.ModuleList()
        self.up_cnns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.gunpools = nn.ModuleList()
        self.l_n = l_n
        feature_dim = dim
        for i in range(self.l_n):
            self.down_cnns.append(CNN(self.channels[i], self.channels[i+1], self.kernal_size[i], drop_p))
            self.down_cnns.append(CNN(self.channels[i+1], self.channels[i+1], self.kernal_size[i], drop_p))
            self.pools.append(nn.MaxPool2d(kernel_size = (1, int(1/self.cs[i])),
                                           stride = (1, int(1/self.cs[i])), return_indices=True))
            feature_dim = int(feature_dim * self.cs[i])
            self.down_gcns.append(GCN(feature_dim, self.gcn_h[i], feature_dim, act, drop_p))
            self.gpools.append(gPool(float(self.ks[i]), feature_dim, drop_p))
            self.up_cnns.append(CNN(self.channels[i+2], self.channels[i+1], self.kernal_size[i+1], drop_p))
            self.gunpools.append(gUnpool())
            self.up_gcns.append(GCN(feature_dim, self.gcn_h[i], feature_dim, act, drop_p))
            self.unpools.append(Unpool(self.cs[i]))
            self.up_cnns.append(CNN(self.channels[i+2], self.channels[i+1], self.kernal_size[i], drop_p))
            self.up_cnns.append(CNN(self.channels[i+1], self.channels[i+1], self.kernal_size[i], drop_p))
        self.bottom_cnn1 = CNN(self.channels[-2], self.channels[-1], self.kernal_size[-1], drop_p)
        self.bottom_cnn2 = CNN(self.channels[-1], self.channels[-1], self.kernal_size[-1], drop_p)
        self.bottom_gcn = GCN(feature_dim, self.gcn_h[-1], feature_dim, act, drop_p)
        self.last_cnn = CNN(self.channels[1], 5, self.kernal_size[0], drop_p)


    def forward(self, g, h):
        adj_ms = []
        findices_list = []
        gindices_list = []
        down_couts = []
        down_gouts = []
        h = torch.unsqueeze(h, 1)
        for i in range(self.l_n):
            h = self.down_cnns[2 * i](h)
            h = self.down_cnns[2 * i + 1](h)
            down_couts.append(h)
            h, findices = self.pools[i](h)
            findices_list.append(findices)
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_gouts.append(h)
            g, h, gindices = self.gpools[i](g, h)
            gindices_list.append(gindices)
        h = self.bottom_cnn1(h)
        h = self.bottom_cnn2(h)
        h = self.bottom_gcn(g, h)
        salient_node = gindices_list[-1]
        salient_spacial = g
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            h = self.up_cnns[3 * up_idx](h)
            g, h = self.gunpools[up_idx](adj_ms[up_idx], h, down_gouts[up_idx], gindices_list[up_idx])
            if i == self.l_n - 1:
                ts_trasaction = g
            h = self.up_gcns[up_idx](g, h)
            h = self.unpools[up_idx](h, findices_list[up_idx], down_couts[up_idx])
            h = self.up_cnns[3 * up_idx + 1](h)
            h = self.up_cnns[3 * up_idx + 2](h)
        h = self.last_cnn(h)
        return h, salient_spacial, salient_node, ts_trasaction

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p = p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        o_hs = []
        for i in range(h.shape[0]):
            h = h.to(torch.float32)
            g = g.to(torch.float32)
            hs = torch.matmul(g[0], h[0])
            o_hs.append(hs)
        h = torch.stack(o_hs)

        h = self.proj1(h)
        h = self.act(h)
        h = self.drop(h)

        o_hs = []
        for i in range(h.shape[0]):
            hs = torch.matmul(g[0], h[0])
            o_hs.append(hs)
        h = torch.stack(o_hs)

        h = self.proj2(h)
        h = self.act(h)
        return h

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_p):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = calculate_same_padding(kernel_size)
        self.conv2d = nn.Conv2d(in_channels, out_channels, stride = 1,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                padding=(calculate_same_padding(self.kernel_size),
                                         calculate_same_padding(self.kernel_size)))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x


class gPool(nn.Module):
    def __init__(self, k, fea_dim, p):
        super(gPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(fea_dim, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze(-1)
        weights = weights.transpose(1, 2)
        weights = self.pool(weights).squeeze(-1)
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class gUnpool(nn.Module):
    def __init__(self):
        super(gUnpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([h.shape[0], h.shape[1], g.shape[1], h.shape[-1]])
        for i in range(h.shape[0]):
            new_h[i][:, idx[i], :] = h[i]
        new_h = new_h.add(pre_h)
        return g, new_h


class Unpool(nn.Module):
    def __init__(self, cs):
        super(Unpool, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size = (1, 1/cs), stride = (1, 1/cs))

    def forward(self, x, indices, pre_x):
        x = self.unpool(x, indices, output_size = pre_x.shape)
        result = torch.cat((x, pre_x), dim = 1)
        return result


def top_k_graph(scores, g, h, k):
    batch_size, num_nodes = g.shape[0], g.shape[1]
    values, idx = torch.topk(scores, max(2, int(float(k)*num_nodes)))
    o_hs = []
    for i in range(batch_size):
        o_hs.append(h[i][:, idx[i], :])
    new_h = torch.stack(o_hs)
    o_hs = []
    for i in range(batch_size):
        value = torch.unsqueeze(values[i], -1)
        o_hs.append(torch.mul(new_h[i], value))
    new_h = torch.stack(o_hs)
    un_gs = []
    for i in range(batch_size):
        un_g = g[i].bool().float()
        un_g = torch.matmul(un_g, un_g).bool().float()
        un_g = un_g[idx[i], :]
        un_g = un_g[:, idx[i]]
        un_gs.append(un_g)
    un_g = torch.stack(un_gs)
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees.unsqueeze(1)
    return g

class Initializer(object):
    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)