import argparse
from itertools import product
from idna import valid_contextj
import torch 
from datasets import get_dataset

import os

from blend import Blend
from train_eval import cross_validation_with_val_set
import datetime
import time as timet
from rd.utils import count_parameters, addLaplacianPE, seed_everywhere

from torch_geometric.utils import get_laplacian, to_dense_adj, to_scipy_sparse_matrix
import torch_geometric.transforms as T
from scipy.sparse.linalg import eigsh, eigs
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.99)
parser.add_argument('--lr_decay_step_size', type=int, default=20)
parser.add_argument('--w_decay', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='PTC_MR')

### Experiment
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension.') 
parser.add_argument('--step_size', type=float, default=0.1, help='fixed step size when using fixed step solvers e.g. rk4')

# Encoder args
parser.add_argument("--encoder_layers", type=int, default=2, help="depth of encoder")

# Reaction-diffusion args
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

parser.add_argument('--data_norm', type=str, default='rw', help='rw for random walk, gcn for symmetric gcn norm')
parser.add_argument('--self_loop_weight', type=float, default=0.0,help='Weight of self-loops.')
 
parser.add_argument('--method', type=str,default='euler', help="set the numerical solver: dopri5, euler, rk4, midpoint, symplectic_euler, leapfrog")
parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
parser.add_argument("--adjoint_method", type=str, default="adaptive_heun", help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
parser.add_argument('--adjoint', dest='adjoint', action='store_true', help='use the adjoint ODE method to reduce memory footprint')
parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
parser.add_argument("--tol_scale_adjoint", type=float, default=1.0, help="multiplier for adjoint_atol and adjoint_rtol")
 
parser.add_argument('--beta_diag', type=eval, default=False)

# Attention args
parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                    help='slope of the negative part of the leaky relu used in attention')
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--mha_heads', type=int, default=8, help='number of mha module heads')
parser.add_argument('--attention_dropout', type=float, default=0.01, help='dropout of attention weights')
parser.add_argument('--mha_dropout', type=float, default=0.01, help='dropout of mha module weights')
parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
parser.add_argument('--attention_dim', type=int, default=128,
                    help='the size to project x to before calculating att scores')
parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                    help='apply a feature transformation xW to the ODE')
parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                    help="multiply attention scores by edge weights before softmax")
parser.add_argument('--attention_type', type=str, default="scaled_dot",
                    help="scaled_dot,cosine_sim,pearson, exp_kernel")
parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')


args = parser.parse_args()
opt = vars(args)
 
datasets = [args.dataset]

nets = [Blend]

seed_everywhere(1993)

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.4f}')

now = datetime.datetime.now()
time = now.strftime("%m-%d-%H:%M")

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    if not os.path.exists('out/'+dataset_name):
        os.makedirs('out/'+dataset_name) 
    print(f'--\n{dataset_name} - {Net.__name__} - {time}')  
        
    dataset = get_dataset(dataset_name)
    model = Net(opt, dataset) 

    from rd.utils import print_model_params
    print_model_params(model)

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

    desc = f'{best_result[1]:.4f}\(\pm\){best_result[2]:.4f} | duration_mean: {best_result[3]/1000}'
        
    results += [f'{dataset_name} - {model}: {desc}, epochs: {opt["epochs"]}']
results = '\n'.join(results)
 
print(f'--\n{results}') 
