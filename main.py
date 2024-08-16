import argparse
from itertools import product
import torch
from asap import ASAP
from datasets import get_dataset
from diff_pool import DiffPool
from edge_pool import EdgePool
import os
from gcn import GCN, GCNWithJK
from gat import GAT
from gin import GIN, GIN0, GIN0WithJK, GINWithJK
from global_attention import GlobalAttentionNet
from graclus import Graclus
from graph_sage import GraphSAGE, GraphSAGEWithJK
from sag_pool import SAGPool
from set2set import Set2SetNet
from sort_pool import SortPool
from top_k import TopK
from blend import Blend, addLaplacianPE
from train_eval import cross_validation_with_val_set
import numpy as np
import random
import torch.backends.cudnn as cudnn
import datetime
import sys
import time as timet
from rd.utils import count_parameters

from torch_geometric.utils import get_laplacian, to_dense_adj, to_scipy_sparse_matrix
import torch_geometric.transforms as T
from scipy.sparse.linalg import eigsh, eigs
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.99)
parser.add_argument('--lr_decay_step_size', type=int, default=20)
parser.add_argument('--w_decay', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='MUTAG') 
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate.')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension.')

# Blending args
parser.add_argument("--lambda", type=float, default=0.5, help="lambda")

# Encoder args
parser.add_argument("--encoder_layers", type=int, default=2, help="depth of encoder")

# GREAD reaction-diffusion args
parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
parser.add_argument('--time', type=float, default=2.0, help='End time of ODE integrator.')
parser.add_argument('--block', type=str, default='attention', help='constant, attention')
parser.add_argument('--function', type=str, default='gread', help='laplacian, transformer, gread, GAT')
parser.add_argument('--reaction_term', type=str, default='bspm', help='bspm, fisher, allen-cahn, zeldovich, st, fb, fb3')

parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")
parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")
 
parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
parser.add_argument('--alpha_dim', type=str, default='vc', help='choose either scalar (sc) or vector (vc) alpha')
parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
parser.add_argument('--source_dim', type=str, default='vc', help='choose either scalar (sc) or vector (vc) source')
parser.add_argument('--beta_dim', type=str, default='vc', help='choose either scalar (sc) or vector (vc) beta')
parser.add_argument('--beta_diag', type=eval, default=False)
 
parser.add_argument('--method', type=str,default='euler', help="set the numerical solver: dopri5, euler, rk4, midpoint, symplectic_euler, leapfrog")
parser.add_argument('--step_size', type=float, default=0.1, help='fixed step size when using fixed step solvers e.g. rk4')
# parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
# parser.add_argument("--adjoint_method", type=str, default="adaptive_heun", help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
parser.add_argument('--adjoint', dest='adjoint', action='store_true', help='use the adjoint ODE method to reduce memory footprint')
# parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
# parser.add_argument("--tol_scale_adjoint", type=float, default=1.0, help="multiplier for adjoint_atol and adjoint_rtol")
 

# GREAD attention args
parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                    help='slope of the negative part of the leaky relu used in attention')
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                    help="multiply attention scores by edge weights before softmax")
parser.add_argument('--attention_type', type=str, default="scaled_dot",
                    help="scaled_dot,cosine_sim,pearson, exp_kernel")
parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

### analysis
parser.add_argument("--ablation_study", type=str, default="none", help="none, only_global, only_local, only_macro, only_micro, no_global, linear, no_blend")
parser.add_argument('--drawing', type=eval, default=False)
parser.add_argument("--pool_op", type=str, default="sum", help="sum, mean")

args = parser.parse_args()
opt = vars(args)

def seed_everywhere(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

####### for baselines #######
layers = [2]
hiddens = [128]
#############################
datasets = [args.dataset]

nets = [
    Blend,
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    # Graclus,
    # TopK,
    # SAGPool,
    # EdgePool,
    # GCN,
    # GraphSAGE,
    # GIN0,
    # GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    # ASAP,
    # DiffPool,
    ]

seed_everywhere(2024)

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')

now = datetime.datetime.now()
time = now.strftime("%m-%d-%H:%M")

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    if not os.path.exists('out/'+dataset_name):
        os.makedirs('out/'+dataset_name)
    print(f'--\n{dataset_name} - {Net.__name__} - {time}')

    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        if opt["ablation_study"] not in ["only_local", "linear", "no_global"] and Net == Blend: 
            if dataset_name in ['REDDIT-BINARY']: 
                dataset.transform = T.Compose([dataset.transform, T.AddLaplacianEigenvectorPE(k=1, attr_name=None, tol=0.02)]) 
            elif dataset_name in ['IMDB-BINARY', 'IMDB-MULTI']: 
                dataset.transform = T.Compose([dataset.transform, T.AddLaplacianEigenvectorPE(k=1, attr_name=None)]) 
            else:
                addLaplacianPE(dataset, dataset_name)

        if Net == Blend:
            model = Net(opt, dataset)
        else:
            model = Net(dataset, num_layers, hidden)
        loss, acc, std, duration_mean = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=args.w_decay, 
            logger=None,
            dataset_name=dataset_name,
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std, duration_mean)

    desc = f'{best_result[1]:.4f} ± {best_result[2]:.4f} | duration_mean: {best_result[3]}'

    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}, epochs: {opt["epochs"]}, hidden_dim: {opt["hidden_dim"]}, time: {opt["time"]}, method: {opt["method"]}, ablation_study: {opt["ablation_study"]}, pool_op: {opt["pool_op"]}, step_size: {opt["step_size"]}, lambda: {opt["lambda"]}']
results = '\n'.join(results)

print(f'--\n{results}')
