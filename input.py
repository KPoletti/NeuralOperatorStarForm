import torch.nn as nn
from utils.dissipative_utils import (
    sample_uniform_spherical_shell,
    linear_scale_dissipative_target,
)
from neuralop.training.losses import LpLoss, H1Loss
import math

# Data parameters
mu = 2
dN = 1
sub = 1
# Pooling parameters
poolKernel = 4  # set to 0 to disable pooling
poolStride = 4  # set to 0 to disable pooling

S = 800
# calculate the size of the output of the pooling layer from
# https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
if poolKernel > 0:
    S = math.floor((S - poolKernel) / poolStride + 1)
T = 1
T_in = 1
split = [0.7, 0.2, 0.1]
DATA_PATH = "../dataToSend/TrainingData/"
TRAIN_PATH = f"{DATA_PATH}density_mu{mu}_dN{dN}.pt"
TIME_PATH = f"{DATA_PATH}time_mu{mu}_dN{dN}.h5"
log = True
N = 874

# DATA_PATH = "../dataToSend/CaltechData/"
# TRAIN_PATH = f"{DATA_PATH}ns_V1e-3_N5000_T50.pt"
# TIME_PATH = f"{DATA_PATH}ns_V1e-3_N5000_T50.h5"
# log = False
# N = 5000

# Neural network parameters
NN = "FNO2d"  # "FNO2d", "MNO", "FNO"
input_channels = 1
output_channels = 1
modes = 10  # star Form 20
width = 120  # star Form 100
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

# Training parameters
epochs = 500
lr = 0.01
scheduler_step = 50
scheduler_gamma = 0.5
batch_size = 10
optimizer = "Adam"
loss_fn = LpLoss(d=2, p=2)
level = "DEBUG"
file = "output.log"
log_interval = 100

# Option to Save NN
saveNeuralNetwork = True

# Option to create plots
doPlot = True
