"""
utility functions
"""
import os
import torch_geometric
import random
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch_geometric.transforms as T
import os.path as osp
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import networkx as nx
def to_networkx_sparse(data):
    G = nx.Graph()

    for i in range(data.num_nodes):
        G.add_node(i)

    edge_index = data.edge_index
    if edge_index is not None:
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

    if data.x is not None:
        for i, feature in enumerate(data.x): 
            G.nodes[i]['label'] = feature.argmax().item()
            # G.nodes[i]['label'] = feature.sum().item()
    G.label = data.y.item()
    return G

def visualize_molecular_structure(G):
    pos = nx.spring_layout(G)

    node_features = nx.get_node_attributes(G, 'label')
    unique_features = list(set([f for f in node_features.values()]))
    color_map = plt.cm.get_cmap('viridis', len(unique_features))
    # color_map = ListedColormap(['red', 'green'])

    node_colors = [color_map(unique_features.index(node_features[node])) for node in G.nodes()]

    nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color='gray', node_size=100, width=1)

    feature_color_mapping = {feature: color_map.colors[idx] for idx, feature in enumerate(unique_features)}
    legend_labels = {f'{feature}': color for feature, color in feature_color_mapping.items()}
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10) for label, color in legend_labels.items()],
               title="Time")
    plt.draw()
    plt.savefig('sample.jpg', dpi=300)
    plt.cla()
    plt.clf()
    # plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class MaxNFEException(Exception): pass


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  return edge_index, edge_weight

# adapted from make symmetric in graph_rewiring.py
def make_symmetric(edge_index, values, n):
  ###test ideas: assert symetric matrix, compare to dense form
  ApAT_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
  ApAT_value = torch.cat([values, values], dim=0) / 2
  ei, ew = coalesce(ApAT_index, ApAT_value, n, n, op="add")
  scatter_add
  return ei, ew


def is_symmetric(index, value, n):
  ###check a sparse tensor is symmetric### #todo check this works
  index_t, value_t = transpose(index, value, n, n)
  i0, v0 = coalesce(index, value, n, n, op="add")
  it, vt = coalesce(index_t, value_t, n, n, op="add")
  assert torch.all(torch.eq(i0, it)), 'edge index is equal'
  assert torch.all(torch.eq(v0, vt)), 'edge index was reordered'


def make_symmetric_unordered(index, value):
  ### takes multiheaded attention and does (A+A.T)/2 but keeps given index ordering
  # is_symmetric(index, value, n) #todo include this
  d = {(index[0, i].item(), index[1, i].item()): value[i] for i in range(index.shape[1])}
  trans = torch.stack([d[(index[1, i].item(), index[0, i].item())] for i in range(index.shape[1])], dim=0)
  return (value + trans) / 2


def sym_row_max(edge_index, values, n):
  row_max = scatter_add(values, edge_index[0], dim=0, dim_size=n).max()
  values = values / row_max
  return values, row_max


def sym_row_col(edge_index, values, n):
  #doesn't need symmetric matrix but can be made more efficient with that assumption
  row_sum = scatter_add(values, edge_index[0], dim=0, dim_size=n)
  col_sum = scatter_add(values, edge_index[1], dim=0, dim_size=n)
  row_sum_sq = torch.pow(row_sum, -0.5)
  col_sum_sq = torch.pow(col_sum, -0.5)
  return row_sum_sq[edge_index[0]] * values * col_sum_sq[edge_index[1]]
  #Assuming symetric
  # row_sum = scatter_add(values, edge_index[0], dim=0, dim_size=n)
  # row_sum_sq = torch.pow(row_sum, -0.5)
  # return row_sum_sq[edge_index[0]] * values * row_sum_sq[edge_index[1]]

def sym_row_col_att(edge_index, A, edges, values, n):
  row_sum = scatter_add(A, edge_index[0], dim=0, dim_size=n)
  col_sum = scatter_add(A, edge_index[1], dim=0, dim_size=n)
  row_sum_sq = torch.pow(row_sum, -0.5)
  col_sum_sq = torch.pow(col_sum, -0.5)
  return row_sum_sq[edges[0]] * values * col_sum_sq[edges[1]]

def sym_row_col_att_measure(edge_index, A, edges, values, measure, n):
  row_sum = scatter_add(A, edge_index[0], dim=0, dim_size=n)
  col_sum = scatter_add(A, edge_index[1], dim=0, dim_size=n)
  row_sum_sq = torch.pow(row_sum * torch.exp(measure), -0.5)
  col_sum_sq = torch.pow(col_sum * torch.exp(measure), -0.5)
  return row_sum_sq[edges[0]] * values * col_sum_sq[edges[1]]

def sym_row_col_att_relaxed(edge_index, A, edges, values, measure, n):
  row_sum = scatter_add(A, edge_index[0], dim=0, dim_size=n)
  col_sum = scatter_add(A, edge_index[1], dim=0, dim_size=n)
  row_sum_sq = torch.pow(row_sum + torch.exp(measure), -0.5)
  col_sum_sq = torch.pow(col_sum + torch.exp(measure), -0.5)
  return row_sum_sq[edges[0]] * values * col_sum_sq[edges[1]]

def sym_row_sum_one():
  # find each
  # LHS @ A @ RHS
  pass

def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index

def dirichlet_energy(edge_index, edge_weight, n, X):
  edge_index, L = get_laplacian(edge_index, edge_weight, None)
  LX = torch_sparse.spmm(edge_index, L, n, n, X)
  # return torch.sum(torch.trace(X.T @ LX))
  return (X * LX).sum()


from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)

def sigmoid_deriv(x):
  return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def tanh_deriv(x):
  return 1 - torch.tanh(x) ** 2

def squareplus_deriv(x):
  return (1 + x / torch.sqrt(x ** 2 + 4)) / 2

# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes


  

def seed_everywhere(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def split_graph(dgl_graph):
    g = []
    y = []
    for graph, label in dgl_graph: 
        g.append(graph)
        y.append(label)

    return g, y

def split_pyg_graph(pyg_graphs):
    g = []
    y = []
    for i in range(0, len(pyg_graphs)):
        g.append(pyg_graphs[i])
        y.append(pyg_graphs[i].y.squeeze())
    return g, y

def dgl_to_pyg(dgl_graph, dgl_y):
    # print(dgl_graph[:])
    pyg = []
    pyg_y = []
    for i in range(0, len(dgl_graph)):
        pyg_g = torch_geometric.utils.from_dgl(dgl_graph[i])
        pyg_g['x'] = pyg_g['attr']
        del pyg_g['attr']
        pyg_g['y'] = dgl_y[i].unsqueeze(0)
        pyg.append(pyg_g)
        pyg_y.append(pyg_g['y'])

    return pyg, pyg_y

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    

def print_results(fold_result_train, fold_result_val):
    print('======== Best Train Results ========')
    res = np.array([list(x.values()) for x in fold_result_train], dtype=float)
    for r in res:
        print('Fold {0:0>1.0f} | epoch: {5:0>1.0f} | Acc : {1:0>4.4f} | Precision: {2:0>4.4f} | Recall: {3:0>4.4f} | Total_oss: {4:0>4.4f} | class_loss: {6:0>4.4f} '.format(r[0], r[1], r[2], r[3], r[4], r[5], r[6] ))
    print('-------- Mean & Std. --------')
    print('Mean Acc : {0:0>4.4f} | Precision: {1:0>4.4f} | Recall: {2:0>4.4f} | Total_loss: {3:0>4.4f} | class_loss {4:0>4.4f} '.format(res[:,1].mean(axis=0), res[:,2].mean(axis=0),res[:,3].mean(axis=0),res[:,4].mean(axis=0), res[:,6].mean(axis=0) ))
    print('Std Acc  : {0:0>4.4f} | Precision: {1:0>4.4f} | Recall: {2:0>4.4f} | Total_loss: {3:0>4.4f} | class_loss {4:0>4.4f} '.format(res[:,1].std(axis=0), res[:,2].std(axis=0),res[:,3].std(axis=0),res[:,4].std(axis=0), res[:,6].std(axis=0) ))

    print('======== Best Validation Results ========')
    res = np.array([list(x.values()) for x in fold_result_val], dtype=float)
    for r in res:
        print('Fold {0:0>1.0f} | epoch: {5:0>1.0f} | Acc : {1:0>4.4f} | Precision: {2:0>4.4f} | Recall: {3:0>4.4f} | Loss: {4:0>4.4f} | class_loss: {6:0>4.4f}  '.format(r[0], r[1], r[2], r[3], r[4], r[5], r[6] ))
    print('-------- Mean & Std. --------')
    print('Mean Acc : {0:0>4.4f} | Precision: {1:0>4.4f} | Recall: {2:0>4.4f} | Total_loss: {3:0>4.4f} | class_loss {4:0>4.4f} '.format(res[:,1].mean(axis=0), res[:,2].mean(axis=0), res[:,3].mean(axis=0), res[:,4].mean(axis=0), res[:,6].mean(axis=0) ))
    print('Std  Acc : {0:0>4.4f} | Precision: {1:0>4.4f} | Recall: {2:0>4.4f} | Total_loss: {3:0>4.4f} | class_loss {4:0>4.4f} '.format(res[:,1].std(axis=0), res[:,2].std(axis=0), res[:,3].std(axis=0), res[:,4].std(axis=0), res[:,6].std(axis=0) ))

def add_labels(feat, labels, idx, num_features, device):
    onehot = torch.zeros([feat.shape[0], num_features]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1

    return torch.cat([feat, onehot], dim=-1)

class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
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


