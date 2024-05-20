from rd.function_laplacian_diffusion import LaplacianODEFunc
from rd.block_constant import ConstantODEblock
from rd.block_attention import AttODEblock
from rd.function_gread import ODEFuncGread

class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass

def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'attention':
    block = AttODEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'gread':
    f = ODEFuncGread
  else:
    raise FunctionNotDefined
  return f