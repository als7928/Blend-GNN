import torch
import torch.nn as nn
import torch.nn.functional as F
from rd.model_configurations import set_block, set_function
from rd.base_classes import BaseGNN
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import numpy as np



class Blending(nn.Module):
    def __init__(self, input_dim, device):
        super(Blending, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.weights = nn.Parameter(torch.randn(1, self.input_dim, device=self.device))
        self._reset_parameters()

    def forward(self, A, B): 
        a_score = F.linear(F.normalize(A, p=2, dim=1), F.normalize(self.weights, p=2, dim=1)).squeeze()
        b_score = F.linear(F.normalize(B, p=2, dim=1), F.normalize(self.weights, p=2, dim=1)).squeeze()        
        scores = torch.stack([a_score, b_score], dim=1)
        
        w = F.softmax(scores, dim=1) 
        # print(w)
        combined = w[:, 0].unsqueeze(1) * A + w[:, 1].unsqueeze(1) * B

        return combined, w
     
    
    def _reset_parameters(self):
        self.weights = nn.Parameter(torch.randn(1, self.input_dim, device=self.device))
        

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
        self.device = device
        self.time_tensor = torch.linspace(0, self.T, 2, dtype=torch.float).to(self.device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.train_dataset._data, self.device, t=self.time_tensor).to(self.device)

        self.odeblock.odefunc.GNN_m2 = self.m2
        
        self.encoder = Encoder(opt, opt["encoder_layers"], self.num_features, opt["hidden_dim"])

        self.graph_wise = Blending(3*opt['hidden_dim'], self.device)  
        self.node_wise = Blending(opt['hidden_dim'], self.device)  

        self.lin1 = torch.nn.Linear(6*opt['hidden_dim'], opt['hidden_dim'])
        self.lin2 = torch.nn.Linear(opt['hidden_dim'], self.num_classes)
        
        self.draw_count = 0 
        self.n_weight = []
        self.g_weight = []
        self.n_weight2 = []
        self.g_weight2 = []
    

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
        
    def get_h0(self, x): 
        return self.encoder(x)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h0 = self.encoder(x)
        hT = self.reactiondiffusion(h0, edge_index=edge_index)
        
        nz, n_weight = self.node_wise(h0, hT) 
        nz = torch.cat([global_mean_pool(nz, batch), global_add_pool(nz, batch), global_max_pool(nz, batch)], dim=1)
        
        g0 = torch.cat([global_mean_pool(h0, batch), global_add_pool(h0, batch), global_max_pool(h0, batch)], dim=1)
        gT = torch.cat([global_mean_pool(hT, batch), global_add_pool(hT, batch), global_max_pool(hT, batch)], dim=1)
        gz, g_weight = self.graph_wise(g0, gT)

        z = torch.cat([nz, gz], dim=1)


       
        z = F.relu(self.lin1(z))
        z = F.dropout(z, p=self.opt['dropout'], training=self.training)
        z = self.lin2(z)
 
        if self.dataset_name in ['Peptides-struct', 'Peptides-func']:
            return z
        else:
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
            
        if hasattr(self, 'graph_wise'): 
            self.graph_wise._reset_parameters() 
        self.lin1.reset_parameters()
        self.lin2.reset_parameters() 