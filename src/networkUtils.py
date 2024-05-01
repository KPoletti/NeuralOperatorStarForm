"""
This module contains utility functions for defining and initializing neural networks
used in the Neural Operator StarForm project.

Functions:
    initializeNetwork(params: dataclass) -> nn.Module:
        Initializes the neural network model based on the input parameters.

Classes:
    None

Exceptions:
    None
"""

import logging
from neuralop.models import FNO2d, FNO3d  # , FNO

import torch
import torch.nn.functional as F
from cliffordlayers.models.basic.twod import (
    CliffordFourierBasicBlock2d,
    CliffordFluidNet2d,
)
from cliffordlayers.models.basic.threed import (
    CliffordFourierBasicBlock3d,
    CliffordMaxwellNet3d,
)
from cliffordlayers.models.utils import partialclass
from torch import nn

logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):

        # encoder
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
        )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        # decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # bottleneck
        bottleneck = self.bottleneck(enc4)
        # decoder
        dec1 = self.upconv1(bottleneck)
        dec1 = self.decoder1(torch.cat((dec1, enc4), dim=1))

        dec2 = self.upconv2(dec1)
        dec2 = self.decoder2(torch.cat((dec2, enc3), dim=1))

        dec3 = self.upconv3(dec2)
        dec3 = self.decoder3(torch.cat((dec3, enc2), dim=1))

        dec4 = self.upconv4(dec3)
        dec4 = self.decoder4(torch.cat((dec4, enc1), dim=1))
        return self.conv(dec4)


# define the network architecture
def initializeNetwork(params) -> nn.Module:
    """
    Initialize the model
    Input:
        params: input parameters from the input file
    Output:
        model: torch.nn.Module
    """
    model = nn.Module()
    # TODO: allow MLP to be input parameter
    if "FNO2d" in params.NN or params.NN == "RNN":
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=params.use_mlp,
            mlp_dropout=params.mlp_dropout,
            preactivation=params.preactivation,
            n_layers=params.n_layers,
            skip=params.skip_type,
            fno_block_precision="mixed",
            stablizer="Tanh",
            factorization=params.factorization,
            rank=params.rank,
            joint_factorization=True,
            norm=params.norm,
        )
    elif params.NN == "FNO3d" or params.NN == "RNN3d":
        model = FNO3d(
            n_modes_depth=params.modes,
            n_modes_width=params.modes,
            n_modes_height=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=params.use_mlp,
            mlp_dropout=params.mlp_dropout,
            preactivation=params.preactivation,
            n_layers=params.n_layers,
            skip=params.skip_type,
            fno_block_precision="mixed",
            stablizer="Tanh",
            factorization=params.factorization,
            rank=params.rank,
            joint_factorization=True,
            norm=params.norm,
        )
    elif params.NN == "MNO":
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=params.use_mlp,
            mlp_dropout=params.mlp_dropout,
            preactivation=params.preactivation,
            n_layers=params.n_layers,
            skip=params.skip_type,
        )
    elif params.NN == "CNL2d":
        model = CliffordFluidNet2d(
            g=params.g,
            block=partialclass(
                "CliffordFourierBasicBlock2d",
                CliffordFourierBasicBlock2d,
                modes1=params.modes,
                modes2=params.modes,
            ),
            num_blocks=[1, 1, 1, 1],
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            hidden_channels=params.width,
            activation=F.gelu,
            norm=params.preactivation,
            rotation=False,
        )
    elif params.NN == "CNL3d":
        model = CliffordMaxwellNet3d(
            g=[1, 1, 1],
            block=partialclass(
                "CliffordFourierBasicBlock3d",
                CliffordFourierBasicBlock3d,
                modes1=params.modes,
                modes2=params.modes,
                modes3=params.modes,
            ),
            num_blocks=[1, 1, 1, 1],
            in_channels=4,
            out_channels=1,
            hidden_channels=params.width,
            activation=F.gelu,
            norm=False,
        )
    elif params.NN == "UNet":
        model = UNet(params.input_channels, params.output_channels)
    logger.debug(model)
    if params.level == "DEBUG":
        print(model)

    return model
