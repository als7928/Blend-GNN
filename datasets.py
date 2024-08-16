from copy import deepcopy
import os.path as osp
from git import Tree

import torch
from ogb.graphproppred import PygGraphPropPredDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, LRGBDataset, MalNetTiny, GNNBenchmarkDataset, ZINC, UPFD
from torch_geometric.utils import degree
from dgl.data import TUDataset as DGL_TU

class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        # data = cat_node_attr(data, deg.view(-1, 1), attr_name=self.attr_name)
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    print(f'Loading {name}')
    if name in ['MUTAG', 'NCI1', 'DD', 'COLLAB', 'ENZYMES', 'PROTEINS', 'PTC_MR', 'REDDIT-BINARY', 'NCI109', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY']:
        dataset = TUDataset(path, name, cleaned=False)
    else:
        print(f'Unknown dataset {name}')
        raise ValueError
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree) 
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std) 
    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes) 
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)]) 

    return dataset
