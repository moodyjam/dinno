import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from util import create_graph
from copy import deepcopy
import torch

class ConvNeuralNet(nn.Module):
	#  Simple Convolutional Neural Net Implementation of DiNNO Architecture
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.fc1 = nn.Linear(576, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# define the LightningModule
class DiNNO(L.LightningModule):
    def __init__(self, agents, graph_type="complete", fiedler_value=None, num_classes=10):
        super().__init__()
        self.graph_type = graph_type
        self.fiedler_value = fiedler_value
        self.num_nodes = len(agents)
        self.G, self.G_connectivity = create_graph(num_nodes = self.num_nodes,
                                                   graph_type = graph_type,
                                                   target_connectivity = fiedler_value)
        
        base_model = ConvNeuralNet(num_classes=num_classes)
        self.agent_to_idx = {agent["id"]: i for i, agent in enumerate(agents)}

        # Initialize the networks for each agent
        self.models = nn.ModuleDict({agent["id"]: deepcopy(base_model) for agent in agents})
        self.agents = agents

        # Initialize the dual variables
        params = torch.nn.utils.parameters_to_vector(base_model.parameters())
        self.register_buffer("duals", torch.stack([torch.zeros_like(params) for agent in agents]))
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        # Communicate
        # FIXME Need to create an agent class
        for agent in self.agents:
            agent_idx = self.agent_to_idx[agent["id"]]
            agent_params_vec = torch.nn.utils.parameters_to_vector(self.models[agent["id"]].parameters())
            for neighbor_idx in self.G.neighbors():

        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        
        