import torch.nn as nn
import torch

class Agent(nn.Module):
     def __init__(self, config, model):
        super(Agent, self).__init__()
        self.config = config
        self.model = model
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.register_buffer("dual", torch.zeros_like(params))
        