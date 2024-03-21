import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from util import create_graph
from copy import deepcopy
import torch
from agent import Agent

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
    def __init__(self, agent_config, graph_type="complete", fiedler_value=None, num_classes=10, B=2, rho=0.5):
        super().__init__()
        self.num_nodes = len(agent_config)
        self.G, self.G_connectivity = create_graph(num_nodes = self.num_nodes,
                                                   graph_type = graph_type,
                                                   target_connectivity = fiedler_value)
        
        base_model = ConvNeuralNet(num_classes=num_classes)
        self.agent_id_to_idx = {agent["id"]: i for i, agent in enumerate(agent_config)}

        # Initialize the networks for each agent
        self.models = nn.ModuleDict({agent["id"]: deepcopy(base_model) for agent in agent_config})
        self.agent_config = agent_config
        self.agents = {agent["id"]: Agent(config=agent_config[i],
                                          model=deepcopy(base_model))
                                          for i, agent in enumerate(self.agent_config)}
        
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        # Reset the variable containing the neighbor parameters
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_agent.reset_neighbor_params()

        # Communicate
        for agent_id in self.agents:
            curr_agent_idx = self.agent_id_to_idx[agent_id]
            curr_agent = self.agents[agent_id]
            curr_agent_params = torch.nn.utils.parameters_to_vector(curr_agent.model.parameters())
            for neighbor_idx in self.G.neighbors(curr_agent_idx):
                neighbor_id = self.agent_config[neighbor_idx]["id"]
                self.agents[neighbor_id].append_neighbor_params(curr_agent_params)

        # Technically this can be done in parallel
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_agent_idx = self.agent_id_to_idx[agent_id]
            theta = torch.nn.utils.parameters_to_vector(curr_agent.model.parameters())
            neighbor_params = torch.stack(curr_agent.neighbor_params)
            dual = curr_agent.dual + self.hparams.rho * (theta - neighbor_params).sum(0)
            primal = theta
            x, y = batch[curr_agent_idx]
            for tau in range(self.hparams.B):
                





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

        
        