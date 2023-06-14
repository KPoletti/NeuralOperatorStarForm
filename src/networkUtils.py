import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO3d, FNO2d, FNO
from cliffordlayers.models.utils import partialclass
from cliffordlayers.models.models_2d import (
    CliffordNet2d,
    CliffordFourierBasicBlock2d,
)
from cliffordlayers.models.models_3d import (
    CliffordNet3d,
    CliffordFourierBasicBlock3d,
)
import input as param
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# define the network architecture
def initializeNetwork(params: dataclass) -> nn.Module:
    """
    Initialize the model
    Input:
        params: input parameters from the input file
    Output:
        model: torch.nn.Module
    """
    models = {
        "FNO3d": FNO3d,
        "MNO": FNO2d,
        "FNO": FNO,
        "CNL2d": CliffordNet2d,
        "CNL3d": CliffordNet3d,
    }
    # TODO: allow MLP to be input parameter
    if params.NN == "FNO2d":
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "FNO3d":
        model = FNO3d(
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
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "CNL2d":
        model = CliffordNet2d(
            g=[-1, -1],
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
            norm=False,
            rotation=False,
        )
    elif params.NN == "CNL3d":
        model = CliffordNet3d(
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
    logger.debug(model)
    return model
