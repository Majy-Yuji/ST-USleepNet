import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def calculate_same_padding(kernel_size):
    # 对于 stride = 1, dilation = 1 的情况
    return (kernel_size - 1) // 2

class GraphUnet(nn.Module):
    def __init__(self, ks, cs, kernal, chs, gcn_h, l_n, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = [float(num) for num in ks.split(" ")]
        self.cs = [float(num) for num in cs.split(" ")]
        self.gcn_h = [int(num) for num in gcn_h.split(" ")]
        self.channels = [1] + [int(num) for num in chs.split(" ")]
        self.kernal_size = kernal
        self.padding = calculate_same_padding(self.kernal_size)
        # Down ModuleLists
        self.down_cnns = nn.ModuleList()
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.gpools = nn.ModuleList()
        # Down ModuleLists
        self.up_cnns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.gunpools = nn.ModuleList()
        self.l_n = l_n
        feature_dim = dim
        for i in range(self.l_n):
            # Down Blocks
            self.down_cnns.append(CNN(self.channels[i], self.channels[i+1], self.kernal_size, drop_p))
            self.down_cnns.append(CNN(self.channels[i+1], self.channels[i+1], self.kernal_size, drop_p))
            self.pools.append(nn.MaxPool2d(kernel_size = (1, int(1/self.cs[i])),
                                           stride = (1, int(1/self.cs[i])), return_indices=True))
            feature_dim = int(feature_dim * self.cs[i])
            self.down_gcns.append(GCN(feature_dim, self.gcn_h[i], feature_dim, act, drop_p))
            self.gpools.append(gPool(float(self.ks[i]), feature_dim, self.channels[i+1], drop_p))
            # Up Blocks
            self.up_cnns.append(CNN(self.channels[i+2], self.channels[i+1], self.kernal_size, drop_p))
            self.gunpools.append(gUnpool())
            self.up_gcns.append(GCN(feature_dim, self.gcn_h[i], feature_dim, act, drop_p))
            self.unpools.append(Unpool(self.cs[i]))
            self.up_cnns.append(CNN(self.channels[i+2], self.channels[i+1], self.kernal_size, drop_p))
            self.up_cnns.append(CNN(self.channels[i+1], self.channels[i+1], self.kernal_size, drop_p))
        # Bottom Block
        self.bottom_cnn1 = CNN(self.channels[-2], self.channels[-1], self.kernal_size, drop_p)
        self.bottom_cnn2 = CNN(self.channels[-1], self.channels[-1], self.kernal_size, drop_p)
        self.bottom_gcn = GCN(feature_dim, self.gcn_h[-1], feature_dim, act, drop_p)
        self.last_cnn = CNN(self.channels[1], self.channels[0], self.kernal_size, drop_p)


    def forward(self, g, h):
        # 记录最原始邻接矩阵的形状（用于 gUnPool）
        adj_ms = []
        findices_list = []
        gindices_list = []
        down_couts = []
        down_gouts = []
        # 加一维通道数
        h = torch.unsqueeze(h, 1)
        # Down Blocks
        for i in range(self.l_n):
            # 两层 CNN
            h = self.down_cnns[2 * i](h)
            h = self.down_cnns[2 * i + 1](h)
            # 将 CNN 的输出存入 down_couts
            down_couts.append(h)
            # 进行 MaxPool
            h, findices = self.pools[i](h)
            # 获取保留下的特征编号
            findices_list.append(findices)
            # GCN
            h = self.down_gcns[i](g, h)
            # 将其邻接矩阵存入 adj_ms
            adj_ms.append(g)
            # 将 GCN 输出存入 down_gouts
            down_gouts.append(h)
            # 进行 gpool
            g, h, gindices = self.gpools[i](g, h)
            # 获取保留下的节点编号
            gindices_list.append(gindices)
        # Bottom Block
        h = self.bottom_cnn1(h)
        h = self.bottom_cnn2(h)
        h = self.bottom_gcn(g, h)
        # Up Blocks
        for i in range(self.l_n):
            # 获取反向 ID
            up_idx = self.l_n - i - 1
            # 先 CNN
            h = self.up_cnns[3 * up_idx](h)
            # 进行 gUnpool + concat
            g, h = self.gunpools[up_idx](adj_ms[up_idx], h, down_gouts[up_idx], gindices_list[up_idx])
            # 使用 GCN
            h = self.up_gcns[up_idx](g, h)
            # Unpool (Unpool + CNN + Concat)
            h = self.unpools[up_idx](h, findices_list[up_idx], down_couts[up_idx])
            h = self.up_cnns[3 * up_idx + 1](h)
            h = self.up_cnns[3 * up_idx + 2](h)
        # TODO: 删除与原始张量相加
        h = self.last_cnn(h)
        return h

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
        # g: [batch_size, node, node]
        # h: [batch_size, channel_size, node, feature]
        h = self.drop(h)
        o_hs = []
        for i in range(h.shape[0]):
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

# CNN + BN + Dropout + ELU
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
    def __init__(self, k, fea_dim, channel_dim, p):
        super(gPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(fea_dim, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        # [batch_size, channel_num, node_dim]
        weights = weights.transpose(0, 1)
        # TODO: 可以尝试其他改动（关于计算 top_k）
        weights = weights.permute(1, 2, 0)
        weights = self.pool(weights).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class gUnpool(nn.Module):
    def __init__(self):
        super(gUnpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([h.shape[0], h.shape[1], g.shape[1], h.shape[-1]])
        for i in range(h.shape[0]):
            new_h[i][:, idx[i], :] = h[i]
        # add weights
        new_h = new_h.add(pre_h)
        return g, new_h


class Unpool(nn.Module):
    def __init__(self, cs):
        super(Unpool, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size = (1, 1/cs), stride = (1, 1/cs))

    def forward(self, x, indices, pre_x):
        x = self.unpool(x, indices, output_size = pre_x.shape)
        # TODO: 从 add 改为 cancat
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


# 对权重进行归一化
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