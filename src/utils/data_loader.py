import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity


class G_data(object):
    def __init__(self, num_class, feat_dim, g_list, args):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.g_list = g_list
        self.seed = args.seed
        self.sep_data()

    def sep_data(self):
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = self.seed)
        labels = [g.label for g in self.g_list]
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx, idx, if_inter):
        self.fold_idx = fold_idx - 1
        if fold_idx == 10:
            train_val_idx, test_idx = self.idx_list[0]
        else:
            train_val_idx, test_idx = self.idx_list[fold_idx]
        train_val_gs = [self.g_list[i] for i in train_val_idx]
        skf2 = StratifiedKFold(n_splits = 9, shuffle = True, random_state = self.seed)
        labels = [g.label for g in train_val_gs]
        self.train_val_list = list(skf2.split(np.zeros(len(labels)), labels))
        if fold_idx == 10:
            train_idx, val_idx = self.train_val_list[0]
        else:
            train_idx, val_idx = self.train_val_list[fold_idx]
        self.train_gs = [train_val_gs[i] for i in train_idx]
        self.val_gs = [train_val_gs[i] for i in val_idx]
        if if_inter:
            self.test_gs = [self.g_list[i] for i in test_idx][idx * 32: (idx+1) * 32]
        else:
            self.test_gs = [self.g_list[i] for i in test_idx]


class FileLoader(object):
    def __init__(self, args):
        self.args = args
        self.delta_t = args.delta_t
        self.delta_p = args.delta_p
        self.num_class = args.num_class
        self.feat_dim = args.feat_dim
        self.num_patch = args.num_patch
        self.num_node = args.num_node

    def line_genor(self, lines):
        for line in lines:
            yield line

    def gen_graph(self, psg, label):
        g = nx.Graph()
        node_features = []
        for j in range(len(psg)):
            g.add_node(j, features = psg[j])
            node_features.append(psg[j])
        similarity_matrix = cosine_similarity(node_features)
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > 0.5:
                    g.add_edge(i, j, weight=similarity_matrix[i, j])
        g.label = label
        return g

    def process_g(self, g):
        node_features = []
        num_node = self.num_node
        for j in range(self.num_patch * num_node):
            node_features.append(g.nodes[j]['features'])
        node_features = np.array(node_features)
        g.feas = torch.tensor(node_features)
        A = torch.FloatTensor(nx.to_numpy_array(g))
        g.A = A + torch.eye(g.number_of_nodes())
        time_matrix = np.zeros((self.num_patch * num_node, self.num_patch * num_node))
        for i in range(self.num_patch):
            for j in range(self.num_patch):
                time_matrix[i * num_node:(i + 1) * num_node,
                j * num_node:(j + 1) * num_node] = self.delta_t** abs(i - j)
        position_matrix = np.zeros((self.num_patch * num_node, self.num_patch * num_node))
        for i in range(0, self.num_patch * num_node):
            for j in range(0, self.num_patch * num_node):
                if i % num_node == j % num_node:
                    position_matrix[i, j] = 1
                else:
                    position_matrix[i, j] = self.delta_p
        g.A = g.A * time_matrix * position_matrix
        return g

    def load_data(self):
        args = self.args
        print('loading data ...')
        g_list = []

        data = np.load('data/%s/%s.npz' % (args.data, args.data), allow_pickle=True)
        datas = data['datas']
        labels = data['labels']

        for i in tqdm(range(len(datas)), desc = "Create graph", unit = 'Graph'):
            g = self.gen_graph(datas[i], labels[i])
            g_list.append(g)
        f_n = partial(self.process_g)
        new_g_list = []
        for g in tqdm(g_list, desc = "Process graph", unit = 'Graph'):
            new_g_list.append(f_n(g))

        return G_data(self.num_class, self.feat_dim, new_g_list, self.args)