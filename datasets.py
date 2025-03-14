import os.path as osp
from matplotlib.cbook import to_filehandle
from pkg_resources import to_filename
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, LRGBDataset, GNNBenchmarkDataset
from torch_geometric.utils import degree
from train_eval import k_fold
from rd.utils import seed_everywhere 
from tqdm import tqdm
from torch_geometric.utils import get_laplacian
import numpy as np
from torch_geometric.utils.convert import to_scipy_sparse_matrix

def to_float(data):
    data.x = data.x.float()
    return data

def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """
    This code has been adapted from  https://arxiv.org/abs/2205.12454
    
    Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

                   
def preprocessing(dataset, name):

    dataset.data.edge_attr = None
    tf_list = []
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            
        if max_degree < 1000:
            transform = T.OneHotDegree(max_degree) 
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            transform = NormalizedDegree(mean, std)
            
        tf_list.append(transform)
    
    tf_list.append(to_float)
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset
    
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

    
class iRunDataset():
    def __init__(self, Dataset, path, name, runs=10, *args, **kwargs):
        self.runs = runs
        self.seed = 2024
        self.Dataset = Dataset 
        
        func = None

        if  self.Dataset in [LRGBDataset, GNNBenchmarkDataset] and func is not None:
            self.train_dataset = self.Dataset(path, split='train', name=name, pre_transform=func)
            self.val_dataset = self.Dataset(path, split='val', name=name, pre_transform=func)
            self.test_dataset = self.Dataset(path, split='test', name=name, pre_transform=func)
        elif self.Dataset in [LRGBDataset, GNNBenchmarkDataset] and func is None:
            self.train_dataset = self.Dataset(path, split='train', name=name)
            self.val_dataset = self.Dataset(path, split='val', name=name)
            self.test_dataset = self.Dataset(path, split='test', name=name)
        elif self.Dataset not in [LRGBDataset, GNNBenchmarkDataset] and func is not None:
            self.train_dataset = self.Dataset(path, split='train', pre_transform=func)
            self.val_dataset = self.Dataset(path, split='val', pre_transform=func)
            self.test_dataset = self.Dataset(path, split='test', pre_transform=func)
        else:
            self.train_dataset = self.Dataset(path, split='train') 
            self.val_dataset = self.Dataset(path, split='val')     
            self.test_dataset = self.Dataset(path, split='test')   
            
        if name != 'none':
            self.train_dataset = preprocessing(self.train_dataset, name)
            self.val_dataset = preprocessing(self.val_dataset, name)
            self.test_dataset = preprocessing(self.test_dataset, name)
        
        
        if name in ['Peptides-struct']:
            self.num_classes = 11
        else:
            self.num_classes = self.train_dataset.num_classes
        
        self.num_features = self.train_dataset.num_features
        
        if hasattr(self.train_dataset,"x"):
            self.num_nodes = self.train_dataset.x.shape[0]
        else:
            self.num_nodes = self.train_dataset.num_nodes
        self.dataset_name = name
        
    def load_dataset(self, index):
        seed_everywhere(self.seed + index)
        return self.train_dataset, self.val_dataset, self.test_dataset
        
    def load_datasets(self):
        for i in range(self.runs):
            yield self.load_dataset(i)
            
class iFCVDataset():
    def __init__(self, Dataset, path, name, folds=10, *args, **kwargs): 
        self.folds = folds  

        self.train_dataset = preprocessing(Dataset(name= name, root= path), name)
        
        self.num_classes = self.train_dataset.num_classes
        self.num_features = self.train_dataset.num_features
        self.dataset_name = name
        
        if hasattr(self.train_dataset,"x"):
            self.num_nodes = self.train_dataset.x.shape[0]
        else:
            self.num_nodes = self.train_dataset.num_nodes
        self.fold_indices = list(zip(*k_fold(self.train_dataset, self.folds, name)))
        
    def load_dataset(self, index):
        train_idx, test_idx, val_idx = self.fold_indices[index]
        
        train_dataset = self.train_dataset[train_idx]
        test_dataset = self.train_dataset[test_idx]
        val_dataset = self.train_dataset[val_idx]
        return train_dataset, test_dataset, val_dataset

    def load_datasets(self):
        for i in range(self.folds):
            yield self.load_dataset(i)

def get_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    print(f'Loading {name}')
    if name in ['DD', 'PROTEINS', 'PTC_MR', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI']:
        dataset = iFCVDataset(TUDataset, path, name, 10) 
    elif name in ['Peptides-func', 'Peptides-struct']:
        dataset = iRunDataset(LRGBDataset, path, name, 1) 
    else:
        print(f'Unknown dataset {name}')
        raise ValueError
            
    return dataset