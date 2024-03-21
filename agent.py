import torch.nn as nn
import torch

class Agent(nn.Module):
   def __init__(self, config, model):
      super(Agent, self).__init__()
      self.config = config
      self.model = model
      params = torch.nn.utils.parameters_to_vector(model.parameters())
      self.register_buffer("dual", torch.zeros_like(params))
      self.neighbor_params = []

   def reset_neighbor_params(self):
      self.neighbor_params = []

   def append_neighbor_params(self, neighbor_params):
      self.neighbor_params.append(neighbor_params)

   
        