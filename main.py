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
from augrd import AugRD
from train_eval import cross_validation_with_val_set
import numpy as np
import random
import torch.backends.cudnn as cudnn
import datetime
from rd.utils import count_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.99)
parser.add_argument('--lr_decay_step_size', type=int, default=20)
parser.add_argument('--w_decay', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='MUTAG')

parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate.')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension.')

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
parser.add_argument('--step_size', type=float, default=0.1, help='fixed step size when using fixed step solvers e.g. rk4')
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
parser.add_argument('--attention_dim', type=int, default=64,
                    help='the size to project x to before calculating att scores')
parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                    help='apply a feature transformation xW to the ODE')
parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                    help="multiply attention scores by edge weights before softmax")
parser.add_argument('--attention_type', type=str, default="scaled_dot",
                    help="scaled_dot,cosine_sim,pearson, exp_kernel")
parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

### analysis
parser.add_argument("--ablation_study", type=str, default="none", help="none, no_u, no_rd, no_r, no_aug, no_pos")
parser.add_argument('--drawing', type=eval, default=False)


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
    AugRD,
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
        
        if opt["ablation_study"]!="no_u":
            import torch_geometric.transforms as T
            if dataset.transform is None:
                dataset.transform = T.AddLaplacianEigenvectorPE(k=1, attr_name=None)
            else:
                dataset.transform = T.Compose([dataset.transform, T.AddLaplacianEigenvectorPE(k=1, attr_name=None)])
            if dataset_name == "REDDIT-BINARY":
                dataset.transform = T.Compose([dataset.transform, T.AddLaplacianEigenvectorPE(k=1, attr_name=None, tol=1e-3)])
                
        if Net == AugRD:
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
    if opt["ablation_study"]=="no_rd":
        params=count_parameters(model.encoder)
    else:
        params=count_parameters(model)
    results += [f'{dataset_name} - {model}: {desc}, params: {params}, time: {opt["time"]}, method: {opt["method"]}, ablation_study: {opt["ablation_study"]}']
results = '\n'.join(results)

print(f'--\n{results}')
