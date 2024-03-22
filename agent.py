import torch.nn as nn
import torch
from torch import optim

class Agent(nn.Module):
   def __init__(self, config, model, lr, idx):
      super(Agent, self).__init__()
      self.config = config
      self.model = model
      self.idx = idx
      self.update_flattened_params()
      self.register_buffer("dual", torch.zeros_like(self.flattened_params))
      self.neighbor_params = []
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

   def reset_neighbor_params(self):
      self.neighbor_params = []
   
   def get_flattened_params(self):
      return self.flattened_params
   
   def get_neighbor_params(self):
      return self.neighbor_params

   def append_neighbor_params(self, neighbor_params):
      self.neighbor_params.append(neighbor_params)

   def update_flattened_params(self):
      self.register_buffer("flattened_params", torch.nn.utils.parameters_to_vector(self.parameters()).clone().detach())

   
        