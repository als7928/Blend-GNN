import torch
import torch.nn as nn
import torch.nn.functional as F
from rd.model_configurations import set_block, set_function
from rd.base_classes import BaseGNN
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_networkx
from rd.utils import to_networkx_sparse, visualize_molecular_structure

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
                    # torch.nn.BatchNorm1d(self.hidden_dim),
                ))
    def forward(self, x):
        z = self.lin_in(x)
        for conv in self.convs:
            z = F.dropout(z + conv(z), p=self.dropout, training=self.training)
        z = self.bn(z)
        return z



class AugRD(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cuda')):
        super(AugRD, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.linspace(0, self.T, 2, dtype=torch.float).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset._data, device, t=time_tensor).to(device)

        self.odeblock.odefunc.GNN_m2 = self.m2
        self.encoder = Encoder(opt, opt["encoder_layers"], self.num_features, opt["hidden_dim"])

        if  self.opt["ablation_study"] in ["no_rd", "no_r", "no_d", "no_aug", "no_pos"] :
            self.lin1 = torch.nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
            if self.opt["ablation_study"] in ["no_pos"]:
                self.att = nn.MultiheadAttention(2*opt['hidden_dim'], opt['mha_heads'], dropout=opt['mha_dropout'])
        else:
            self.att = nn.MultiheadAttention(2*opt['hidden_dim'], opt['mha_heads'], dropout=opt['mha_dropout'])
            self.lin1 = torch.nn.Linear(2*opt['hidden_dim'], opt['hidden_dim'])
            self.h0_PE = torch.zeros(size=(self.num_nodes,opt["hidden_dim"]), device=device)
            self.h1_PE = torch.ones(size=(self.num_nodes,opt["hidden_dim"]), device=device)
            self.h0_PE.requires_grad = False
            self.h1_PE.requires_grad = False
    
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
        num_nodes = x.shape[0]
        h0 = self.encoder(x)
        if self.opt["ablation_study"] == "no_rd":
            z = h0
        else:
            hT = self.reactiondiffusion(h0, edge_index=edge_index)
            if self.opt["ablation_study"] in ["no_aug", "no_r", "no_d"]:
                z = hT
            else:
                if self.opt["ablation_study"]=="no_pos":
                    H = torch.stack([h0,hT], dim=0)
                else:
                    h0_pe = torch.cat([h0, self.h0_PE[:num_nodes]], dim=1)
                    hT_pe = torch.cat([hT, self.h1_PE[:num_nodes]], dim=1)
                    H = torch.stack([h0_pe,hT_pe], dim=0)

                if self.training and self.opt["drawing"]:
                    h_blended, z_weight = self.att(H,H,H, need_weights=True)
                else:
                    h_blended, _ = self.att(H,H,H, need_weights=False)
                z = h_blended.max(dim=0).values

                if self.training and self.opt["drawing"]:
                    z_arg = torch.argmax(z_weight, dim=1).t()[0] # -> [nodes]
                    if self.draw_count % 50 == 0:
                        vis_sample = data[0]
                        vis_sample.x = z_arg[:vis_sample.x.shape[0]].unsqueeze(1)
                        # vis_sample.x = hT[:vis_sample.x.shape[0],:]
                        # vis_sample.x = z[:vis_sample.x.shape[0],:]
                        G = to_networkx_sparse(vis_sample)
                        visualize_molecular_structure(G)
                        self.draw_count=0
                    self.draw_count+=1

        if self.training and self.opt["drawing"] and self.opt["ablation_study"] in ["no_rd", "no_r", "no_d", "no_aug"]:
            if self.draw_count % 50 == 0:
                vis_sample = data[0]
                # vis_sample.x = z[:vis_sample.x.shape[0],:]
                G = to_networkx_sparse(vis_sample)
                visualize_molecular_structure(G)
                self.draw_count=0
            self.draw_count+=1

        z = global_mean_pool(z, batch)
        z = F.relu(self.lin1(z))
        z = F.dropout(z, p=self.opt['dropout'], training=self.training)
        z = self.lin2(z)
        return F.log_softmax(z, dim=-1)