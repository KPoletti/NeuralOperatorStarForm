import torch.nn as nn
from utils.dissipative_utils import (
    sample_uniform_spherical_shell,
    linear_scale_dissipative_target,
)
from neuralop.training.losses import LpLoss, H1Loss

mu = 2
dN = 1
sub = 2
S = 800 // sub
T = 1
T_in = 1
split = [0.7, 0.2, 0.1]
DATA_PATH = "../dataToSend/TrainingData/"
TRAIN_PATH = f"{DATA_PATH}density_mu{mu}_dN{dN}.pt"
TIME_PATH = f"{DATA_PATH}time_mu{mu}_dN{dN}.h5"

# Neural network parameters
NN = "MNO"
input_channels = 4
output_channels = 1
modes = 16
width = 32
encoder = True
if NN == "MNO":
    out_dim = 1
    dissloss = nn.MSELoss(reduction="mean")
    sampling_fn = sample_uniform_spherical_shell
    target_fn = linear_scale_dissipative_target
    radius = 156.25 * S  # radius of inner ball
    scale_down = 0.1  # rate at which to linearly scale down inputs
    loss_weight = 0.01 * S**2  # normalized by L2 norm in function space
    radii = (
        radius,
        (525 * S) + radius,
    )  # inner and outer radii, in L2 norm of function space


epochs = 2
lr = 0.001
scheduler_step = 20
scheduler_gamma = 0.5
batch_size = 10
optimizer = "Adam"
loss_fn = H1Loss()
level = "INFO"
file = "output.log"
