import torch
import torch.nn as nn
import torch.nn.functional as F
from rd.model_configurations import set_block, set_function
from rd.base_classes import BaseGNN
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_networkx, get_laplacian
from rd.utils import to_networkx_sparse, visualize_molecular_structure
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import numpy as np


class Blending(nn.Module):
    def __init__(self, input_dim):
        super(Blending, self).__init__()
        self.input_dim = input_dim
        self.node_wiseention_weights = nn.Parameter(torch.empty(self.input_dim))
        self._reset_parameters()

    def forward(self, A, B):
        a_score = torch.sum(A * self.node_wiseention_weights, dim=1)
        b_score = torch.sum(B * self.node_wiseention_weights, dim=1)
        
        scores = torch.stack([a_score, b_score], dim=1)
        weights = F.softmax(scores, dim=1)
        
        combined = weights[:, 0].unsqueeze(1) * A + weights[:, 1].unsqueeze(1) * B
        
        return combined, weights
    
    def _reset_parameters(self) -> None:
        torch.nn.init.normal_(self.node_wiseention_weights, mean=0, std=1)

class Encoder(nn.Module):
    def __init__(self, opt, layers, in_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.opt = opt
        self.dropout = self.opt['dropout']
        self.layers = layers
        self.hidden_dim = opt['hidden_dim']
        self.convs = torch.nn.ModuleList()
        self.lin_in = torch.nn.Linear(in_dim, hidden_dim)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        for i in range(self.layers - 1):
            self.convs.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                ))
    def forward(self, x):
        z = self.lin_in(x)
        for conv in self.convs:
            z = F.dropout(z + conv(z), p=self.dropout, training=self.training)
        z = self.bn(z)
        return z


class Blend(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cuda')):
        super(Blend, self).__init__(opt, dataset, device) 
        self.f = set_function(opt)
        block = set_block(opt)
        # time_tensor = torch.linspace(0,  self.T, opt['seq_len'], dtype=torch.float).to(device)
        self.lmbda = opt["lambda"]
        time_tensor = torch.linspace(0, self.T, 2, dtype=torch.float).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset._data, device, t=time_tensor).to(device)

        self.odeblock.odefunc.GNN_m2 = self.m2
        self.encoder = Encoder(opt, opt["encoder_layers"], self.num_features, opt["hidden_dim"])

        if self.opt["ablation_study"] in ["only_micro"]:
            self.node_wise = Blending(opt['hidden_dim'])
        elif self.opt["ablation_study"] in ["only_macro"] :
            self.graph_wise = Blending(opt['hidden_dim'])
        elif self.opt["ablation_study"] not in ["only_global", "only_local", "no_blend"]:
            self.node_wise = Blending(opt['hidden_dim'])
            self.graph_wise = Blending(opt['hidden_dim'])
        self.lin1 = torch.nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
        self.lin2 = torch.nn.Linear(opt['hidden_dim'], self.num_classes)
        self.draw_count = 0

    def reactiondiffusion(self, x, edge_index=None):
        self.odeblock.set_x0(x)

        if self.opt['function']=='gread':
            if self.opt['beta_diag'] == True:
                self.odeblock.odefunc.Beta = self.odeblock.odefunc.set_Beta()

        if self.training and self.odeblock.nreg > 0:
            z, self.reg_states = self.odeblock(x, edge_index)
            return z
        else:
            z = self.odeblock(x, edge_index)
            return z

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h0 = self.encoder(x)
        
        if self.opt["ablation_study"] in ["only_global"]:
            z = h0 # [nodes, hidden_dim]
        else:
            hT = self.reactiondiffusion(h0, edge_index=edge_index)
            if self.opt["ablation_study"] in ["only_local", "no_blend"]:
                z = hT
            else:
                if self.opt["ablation_study"] not in ["only_macro"]:
                    z, z_weight = self.node_wise(h0, hT) # node-wise
                
                if self.opt["ablation_study"] not in ["only_micro"]:
                    g0 = global_add_pool(h0, batch) 
                    gT = global_add_pool(hT, batch)
                    gz, g_weight = self.graph_wise(g0, gT)  # graph-wise

                if self.training and self.opt["drawing"]:
                    _, z_arg = torch.max(z_weight, dim=1)                                                                                                                         

                    if self.draw_count % 10 == 0:
                        vis_sample = data[0]
                        vis_sample.x = vis_sample.x[:,:-1]
                        G = to_networkx_sparse(vis_sample, mode=0) # 0: argmax 1: sum
                        visualize_molecular_structure(G, 'sample_ori.pdf',mode=1) # 0: legend

                        vis_sample = data[0]
                        vis_sample.x = vis_sample.x[:,-1]
                        G = to_networkx_sparse(vis_sample, mode=1) 
                        visualize_molecular_structure(G, 'sample_eig.pdf', mode=1) 

                        vis_sample.x = z[:vis_sample.x.shape[0],:]
                        G = to_networkx_sparse(vis_sample, mode=1)
                        visualize_molecular_structure(G, 'sample_z.pdf', mode=1)

                        vis_sample.x = hT[:vis_sample.x.shape[0],:]
                        G = to_networkx_sparse(vis_sample, mode=1)
                        visualize_molecular_structure(G, 'sample_hT.pdf', mode=1)

                        vis_sample.x = z_arg[:vis_sample.x.shape[0]].unsqueeze(1)
                        G = to_networkx_sparse(vis_sample, mode=1)
                        visualize_molecular_structure(G, 'sample_bi.pdf', mode=1)
                        self.draw_count=0
                    self.draw_count+=1
        
        if self.opt["pool_op"]=="sum" and self.opt["ablation_study"] not in ["only_macro"]:
            z = global_add_pool(z, batch)
        elif self.opt["pool_op"]=="mean" and self.opt["ablation_study"] not in ["only_macro"]: 
            z = global_mean_pool(z, batch)

        if self.opt["ablation_study"] in ["only_macro"]:
            z = gz
        elif self.opt["ablation_study"] in ["none"]:
            z = self.lmbda * z + (1-self.lmbda) * gz
        else:
            z = z
        z = F.relu(self.lin1(z))
        z = F.dropout(z, p=self.opt['dropout'], training=self.training)
        z = self.lin2(z)
        return F.log_softmax(z, dim=-1)

    def reset_parameters(self):
        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.odeblock.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            if hasattr(layer, 'reset'):
                layer.reset()
        if hasattr(self, 'node_wise'):
            self.node_wise._reset_parameters()
        if hasattr(self, 'graph_wise'): 
            self.graph_wise._reset_parameters() 
        self.lin1.reset_parameters()
        self.lin2.reset_parameters() 

def addLaplacianPE(dataset, dataset_name, tol=0):
  all_eigenvectors = []
  k = 1
  is_undirected = True
  for data in dataset:
      edge_index = data.edge_index
      num_nodes = data.num_nodes

      assert data.edge_index is not None
      assert num_nodes is not None
      
      edge_index, edge_weight = get_laplacian(
          data.edge_index,
          data.edge_weight,
          normalization='sym',
          num_nodes=num_nodes,
      )

      L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

      if num_nodes < 100:
          from numpy.linalg import eig, eigh
          eig_fn = eig if not is_undirected else eigh

          eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
      else:
          from scipy.sparse.linalg import eigs, eigsh
          eig_fn = eigs if not is_undirected else eigsh

          eig_vals, eig_vecs = eig_fn(  # type: ignore
              L,
              k=k + 1,
              which='SR' if not is_undirected else 'SA',
              return_eigenvectors=True,
              tol = tol
          )

      eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()]) 
      pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
      sign = -1 + 2 * torch.randint(0, 2, (k, ))
      pe *= sign

      all_eigenvectors.append(pe)
  all_eigenvectors_tensor = torch.vstack(all_eigenvectors)
  dataset.data.x = torch.cat([dataset.data.x, all_eigenvectors_tensor], dim=1)