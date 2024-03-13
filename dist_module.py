import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from models import initialize_model
from copy import deepcopy

# define any number of nn.Modules (or use your current ones)
base_model = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

# define the LightningModule
class DistModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.base_model = initialize_model(config["model_type"])
        self.models = {i: deepcopy(model) for i in range(config["num_models"])}
        
        if config["optimizer"] == "dinno":
            optimizer = DiNNO()
        else:
            raise NotImplementedError(f"{config["optimizer"]} not implemented yet.")
            

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
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


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
        
        