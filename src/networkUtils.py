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


from neuralop.models import FNO2d, FNO3d, UNO

import torch
from torch import nn

logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class UNet2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=64):

        # encoder
        super(UNet2d, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.Tanh(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.Tanh(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.Tanh(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.Tanh(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.Tanh(),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.Tanh(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.Tanh(),
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.Tanh(),
        )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(width * 8, width * 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(width * 16, width * 16, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        # decoder
        self.upconv1 = nn.ConvTranspose2d(
            width * 16, width * 8, kernel_size=2, stride=2
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(width * 16, width * 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.upconv2 = nn.ConvTranspose2d(width * 8, width * 4, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(width * 8, width * 4, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.upconv3 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(width * 4, width * 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.upconv4 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.conv = nn.Conv2d(width, out_channels, kernel_size=1)

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
        return FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            projection_channels=params.dim_high,
            lifting_channels=params.dim_high,
            use_mlp=params.use_mlp,
            mlp_dropout=params.mlp_dropout,
            preactivation=params.preactivation,
            n_layers=params.n_layers,
            skip=params.skip_type,
            fno_block_precision="mixed",
            stablizer="tanh",
            factorization=params.factorization,
            positional_embedding=None,
            rank=params.rank,
            joint_factorization=True,
            norm=params.norm,
        )
    elif params.NN == "FNO3d" or params.NN == "RNN3d":
        return FNO(
            n_modes=(params.modes,params.modes,params.modes),
            hidden_channels=params.width,
            projection_channel_ratio=params.dim_high,
            lifting_channel_ratio=params.dim_high,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=params.use_mlp,
            mlp_dropout=params.mlp_dropout,
            preactivation=params.preactivation,
            n_layers=params.n_layers,
            skip=params.skip_type,
            fno_block_precision="mixed",
            stablizer="tanh",
            factorization=params.factorization,
            rank=params.rank,
            joint_factorization=True,
            norm=params.norm,
            complex_data=False,
        )
    elif params.NN == "MNO":
        return FNO2d(
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
    elif params.NN == "UNet":
        return UNet2d(params.input_channels, params.output_channels, params.width)
    elif params.NN == "UNO":
        powers = [i for i in range(params.n_layers // 2)]
        powers = powers + powers[::-1]
        if params.n_layers % 2 == 1:
            powers.insert(params.n_layers // 2, params.n_layers // 2)
        scalings = [[1, 1, 1]] * params.n_layers
        scalings = [[1, 1, 1]] * params.n_layers
        if params.n_layers > 2:
            scalings[1] = [0.5, 0.5, 0.5]
            scalings[-1] = [2, 2, 2]
            print([params.width * 2**p for p in powers])
            return UNO(
                in_channels=params.input_channels,
                out_channels=params.output_channels,
                projection_channels=params.width*params.dim_high,
                lifting_channels=params.width*params.dim_high,
                uno_n_modes=[[params.modes] * params.d] * params.n_layers,
                uno_out_channels=[params.width * 2**p for p in powers],
                uno_scalings=scalings,
                hidden_channels=params.width,
                n_layers=params.n_layers,
                channel_mlp_skip="linear",
                mlp_dropout=params.mlp_dropout,
                preactivation=params.preactivation,
                skip=params.skip_type,
                complex_data=False,
                fno_block_precision="mixed", 
            )
        elif params.n_layers == 3:
            m = params.modes
            w = params.width
            return UNO(
                in_channels=params.input_channels,
                out_channels=params.output_channels,
                projection_channels=params.width*params.dim_high,
                lifting_channels=params.width*params.dim_high,
                uno_n_modes=[[m, m, m], [m // 2, m // 2, m // 2], [m, m, m]],
                uno_out_channels=[w, w * 3 // 2, w],
                uno_scalings=scalings,
                hidden_channels=params.width,
                n_layers=params.n_layers,
                channel_mlp_skip="linear",
                mlp_dropout=params.mlp_dropout,
                preactivation=params.preactivation,
                skip=params.skip_type,
                fno_block_precision="mixed",
                complex_data=False,
            )

    return model
