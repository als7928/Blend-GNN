import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from rd.utils import Meter
from rd.regularized_ODE_function import RegularizedODEfunc
import rd.regularized_ODE_function as reg_lib
import six
import torch.nn.functional as F


REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}

def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs


class ODEblock(nn.Module):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t

    self.aug_dim = 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    
    self.nreg = len(regularization_fns)
    self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    self.reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ODEFunc(MessagePassing):
  def __init__(self, opt, data, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    if opt['alpha_dim'] == 'sc':
      self.alpha_train = nn.Parameter(torch.tensor(0.0))
    elif opt['alpha_dim'] == 'vc':
      self.alpha_train = nn.Parameter(0.0*torch.ones(1,opt['hidden_dim']))
    if opt['source_dim'] == 'sc':
      self.source_train = nn.Parameter(torch.tensor(0.0))
    elif opt['source_dim'] == 'vc':
      self.source_train = nn.Parameter(0.0*torch.ones(1,opt['hidden_dim']))
    if opt['beta_dim'] == 'sc':
      self.beta_train = nn.Parameter(torch.tensor(0.0))
    elif opt['beta_dim'] == 'vc':
      self.beta_train = nn.Parameter(0.0*torch.ones(1,opt['hidden_dim']))
       
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.source_sc = nn.Parameter(torch.ones(1))

  def __repr__(self):
    return self.__class__.__name__


class BaseGNN(MessagePassing):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(BaseGNN, self).__init__()
    self.opt = opt
    self.T = opt['time']
    self.num_classes = dataset.num_classes
    self.num_features = dataset.num_features
    self.num_nodes = dataset.num_nodes 
    self.dataset_name = dataset.dataset_name
    
    self.device = device
    self.fm = Meter()
    self.bm = Meter()


    # self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])
    # self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    # self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self.m2 = M2_MLP(opt, dataset, device=device)
  


    # self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
    # self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])

    self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

  def getNFE(self):
    return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def __repr__(self):
    return self.__class__.__name__
  
  def compute_energy(self,x,edge_index):
    energy = self.propagate(edge_index, x=x,energy = True)
    return torch.mean(energy,dim=0).item()

  def message(self, x_i,x_j,energy):
    # x_j has shape [E, out_channels]
    # Step 4: Normalize node features.
    #[E,1]*[E,channel]
    if energy:
        return (torch.linalg.norm(x_j-x_i, dim=1)**2).unsqueeze(dim=1)

  def compute_enegry_evolution(self, edge_index, t_list):
    inter_step = self.odeblock.integrateAt(t_list)
    energy_list = [self.compute_energy(inter_step[i], edge_index) for i in range(inter_step.shape[0])]
    return energy_list

class M2_MLP(nn.Module):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super().__init__()
    self.opt = opt
    self.m21 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self.m22 = nn.Linear(opt['hidden_dim'], dataset.num_features)
    # self.m22 = nn.Linear(opt['hidden_dim'], dataset._data.y.size()[0])
 
  def forward(self, x):
    x = F.dropout(x, self.opt['dropout'], training=self.training)
    x = F.dropout(x + self.m21(torch.tanh(x)), self.opt['dropout'], training=self.training)  # tanh not relu to keep sign, with skip connection
    x = F.dropout(self.m22(torch.tanh(x)), self.opt['dropout'], training=self.training)

    return x