import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedKFold
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity


class G_data(object):
    def __init__(self, num_class, feat_dim, g_list):
        self.num_class = num_class
        # feture_dim = 3000
        self.feat_dim = feat_dim
        # g_list 是图列表
        self.g_list = g_list
        self.sep_data()

    # 划分数据（默认为十折）
    def sep_data(self, seed = 0):
        # 生成一个 StratifiedKFold 实例
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
        # 提取所有 Graph 的标签
        labels = [g.label for g in self.g_list]
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    # 选定特定折的数据
    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx + 1
        train_idx, test_idx = self.idx_list[fold_idx]
        # 在这里调整 train 和 test
        self.train_gs = [self.g_list[i] for i in train_idx]
        self.test_gs = [self.g_list[i] for i in test_idx]


class FileLoader(object):
    def __init__(self, args):
        self.args = args

    def line_genor(self, lines):
        for line in lines:
            yield line

    def gen_graph(self, f):
        row = next(f).strip().split()
        n, label = [int(w) for w in row]
        g = nx.Graph()

        node_features = []
        for j in range(n):
            row = np.array(next(f).strip().split(), dtype = float)
            g.add_node(j, features = row)
            node_features.append(row)
            # TODO 每个节点的 Label 暂时与图的 Label 保持一致
        # 可以调节的超参数
        threshold = 0.5
        similarity_matrix = cosine_similarity(node_features)
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > threshold:
                    g.add_edge(i, j, weight=similarity_matrix[i, j])
        # 整个 graph 的 label
        g.label = label
        # 返回一个生成的图
        return g

    def process_g(self, g):
        # 映射整个 Graph 的 label
        # 映射图节点的 feature 并修改为 one_hot 编码
        node_features = []
        for j in range(10):
            node_features.append(g.nodes[j]['features'])
        node_features = np.array(node_features)
        g.feas = torch.tensor(node_features)
        # 将 Graph 转为一个 numpy 数组
        A = torch.FloatTensor(nx.to_numpy_array(g))
        # 把单位矩阵加到临界矩阵上
        # TODO 同理，也可以加其他种类的边，比如 distance adjacency matrix
        g.A = A + torch.eye(g.number_of_nodes())
        return g

    def load_data(self):
        args = self.args
        print('loading data ...')
        # 是一些 nx.Graph
        g_list = []

        # 读取 data
        with open('data/%s/%s.txt' % (args.data, args.data), 'r') as f:
            lines = f.readlines()
        # 读取第一行
        f = self.line_genor(lines)
        # n_g 为该数据中一共有多少个 Graph
        n_g = int(next(f).strip())
        # n_g = 100
        # 依次读取每一个 Graph 的数据，并形成 Create graph 进度条
        # desc 是在进度条最开始显示的
        # unit 是显示的速度的单位
        for i in tqdm(range(n_g), desc = "Create graph", unit = 'graphs'):
            g = self.gen_graph(f)
            g_list.append(g)

        # 创建一个 f_n 函数
        f_n = partial(self.process_g)
        new_g_list = []
        # 对于 g_list 中的每个图，进行 process(给邻接矩阵加上单位矩阵)
        for g in tqdm(g_list, desc="Process graph", unit='graphs'):
            new_g_list.append(f_n(g))
        num_class = 5
        feat_dim = 3000

        print('# classes: %d' % num_class, '# maximum node tag: %d' % feat_dim)
        return G_data(num_class, feat_dim, new_g_list)