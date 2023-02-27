import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import TFNO3d, TFNO2d, TFNO
from neuralop.training.losses import LpLoss, H1Loss
from dataclasses import dataclass
import numpy as np
import input as param
import logging

logger = logging.getLogger(__name__)

# grab the ending of the density file with .pt
densityProfile = param.TRAIN_PATH.split("density")[-1].split(".")[0]

# create output folder
path = f"SF_{param.NN}_V100_ep{param.epochs}_m{param.modes}_w{param.width}_S{param.S}{densityProfile}"
if not os.path.exists(f"results/{path}"):
    os.makedirs(f"results/{path}")
logger.setLevel(param.level)
fileHandler = logging.FileHandler(f"results/{path}/network.log", mode="w")
formatter = logging.Formatter(
    "%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s"
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# define the network architecture


def initializeNetwork(params: dataclass) -> nn.Module:
    """
    Initialize the model
    Input:
        params: input parameters from the input file
    Output:
        model: torch.nn.Module
    """
    models = {"FNO3d": TFNO3d, "MNO": TFNO2d, "FNO": TFNO}
    # TODO: allow MLP to be input parameter
    if params.NN == "FNO3d":
        model = TFNO3d(
            n_modes_depth=params.modes,
            n_modes_width=params.modes,
            n_modes_height=2,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "MNO":
        model = TFNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    logger.debug(model)
    return model


class Trainer(object):
    def __init__(
        self, model: nn.Module, params: dataclass, device: torch.device
    ) -> None:
        self.model = model
        self.params = params
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=params.scheduler_step,
            gamma=params.scheduler_gamma,
        )
        self.loss = params.loss_fn

    def dissipativeRegularizer(
        self, x: torch.Tensor, device: torch.device
    ) -> torch.nn.MSELoss:
        # TODO: Test this function

        # DISSIPATIVE REGULARIZATION
        x_diss = torch.tensor(
            self.params.sampling_fn(x.shape[0], self.params.radii, x.shape[1:]),
            dtype=torch.float,
        ).to(device)
        assert x_diss.shape == x.shape
        y_diss = self.params.target_fn(x_diss, self.params.scale_down).to(device)
        out_diss = self.model(x_diss).reshape(-1, self.params.out_dim)
        diss_loss = (
            (1 / (self.params.S**2))
            * self.params.loss_weight
            * self.params.dissloss(out_diss, y_diss.reshape(-1, self.params.out_dim))
        )  # weighted by 1 / (S**2)
        return diss_loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        output_encoder,
        regularizer=False,
    ) -> None:
        """
        Train the model
        Input:
            train_loader: torch.utils.data.DataLoader
            valid_loader: torch.utils.data.DataLoader
        Output:
            None
        """
        # initialize the loss function
        self.loss = self.params.loss_fn

        # initialize the regularizer
        # start training
        for epoch in range(self.params.epochs):
            self.model.train()
            loss = 0
            for batch_idx, sample in enumerate(train_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data).view(
                    self.params.batch_size,
                    self.params.S,
                    self.params.S,
                    self.params.T,
                )

                # decode the target
                target = output_encoder.decode(target)
                # decode the output
                output = output_encoder.decode(output)
                #
                loss = self.loss(
                    output.view(self.params.batch_size, -1),
                    target.view(self.params.batch_size, -1),
                )
                if regularizer:
                    loss += self.dissipativeRegularizer(data, self.device)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.params.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
            with torch.no_grad():
                self.model.eval()
                test_loss = 0
                for data, target in test_loader:
                    data, target = sample["x"].to(self.device), sample["y"].to(
                        self.device
                    )
                    output = self.model(data).view(
                        self.params.batch_size,
                        self.params.S,
                        self.params.S,
                        self.params.T,
                    )
                    test_loss += self.loss(
                        output.view(self.params.batch_size, -1),
                        target.view(self.params.batch_size, -1),
                    ).item()
                test_loss /= len(test_loader.dataset)
                print("Test set: Average loss: {:.4f}".format(test_loss))
