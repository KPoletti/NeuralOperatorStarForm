import torch.nn as nn
from src.dissipative_utils import (
    sample_uniform_spherical_shell,
    linear_scale_dissipative_target,
)
from neuralop import LpLoss, H1Loss  # type: ignore
import math

level = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
data_name = "GravColl"  # NS-Caltech, StarForm, GravColl, GravInts or CATS
loss_name = "H1Loss"  # LpLoss, H1Loss

log = False  # Option to take the log of the data
doPlot = True  # Option to create plots
use_mlp = True  # Option to use MLP
encoder = False  # Option to use the encoder
use_ddp = False  # Option to use multi-gpu distributed data
preactivation = True  # Option to use ResNet Preactivation
saveNeuralNetwork = False  # Option to save the neural network
positional_encoding = False
if positional_encoding:
    grid_boundaries = [[-1.25, 1.25], [-1.25, 1.25]]
n_layers = 4
mlp_dropout = 0.01
# Pooling parameters
poolKernel = 0  # set to 0 to disable pooling
poolStride = 0  # set to 0 to disable pooling
#############################################
# FOR NS Cal-tech Data
#############################################
if data_name == "NS-Caltech":
    DATA_PATH = "../dataToSend/CaltechData/"
    TRAIN_PATH = f"{DATA_PATH}ns_V1e-3_N5000_T50.pt"
    TIME_PATH = f"{DATA_PATH}ns_V1e-3_N5000_T50.h5"

    S = 64
    N = 5000

    modes = 12  # star Form 20
    width = 24  # star Form 100
    input_channels = 10
    output_channels = 10

    T = 10
    T_in = 10
    poolKernel = 0  # set to 0 to disable pooling
    poolStride = 0  # set to 0 to disable pooling

##############################################
# For Grav Collapse
##############################################
elif data_name == "GravColl":
    S = 128
    T = 9
    T_in = 1
    DATA_PATH = "../dataToSend/TrainingData/LowRes/"
    dN = 10
    mass = "ALL"
    # dt = 0.0204
    extras = "_multiData"

    TRAIN_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}{extras}.pt"
    TIME_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}{extras}.h5"
    N = 29
    if mass == "ALL":
        N = 902
    data_name = f"{data_name}{mass}_dN{dN}"
    input_channels = 3
    output_channels = 3
    modes = 4  # star Form 20
    width = 10  # star Form 100
    poolKernel = 2  # set to 0 to disable pooling
    poolStride = 2  # set to 0 to disable pooling

elif data_name == "Turb":
    N = 4664
    S = 64
    dN = 10
    T = 5
    T_in = 5

    mass = "12ALL"
    extras = "_Seed"
    DATA_PATH = "../dataToSend/TrainingData/TurbSeed/"
    TRAIN_PATH = f"{DATA_PATH}Turb_VP{mass}_dN{dN}{extras}.pt"
    TIME_PATH = f"{DATA_PATH}Turb_VP{mass}_dN{dN}{extras}.h5"
    # mass = "1-4"
    # extras = "_SmallRange"
    # DATA_PATH = "../dataToSend/TrainingData/TurbProj_SmallerVP/"
    # TRAIN_PATH = f"{DATA_PATH}Turb_VP{mass}_dN{dN}{extras}.pt"
    # TIME_PATH = f"{DATA_PATH}Turb_VP{mass}_dN{dN}{extras}.h5"
    data_name = f"{data_name}_VP{mass}_dN{dN}"

    modes = 8  # star Form 20
    width = 16  # star Form 100
    poolKernel = 0  # set to 0 to disable pooling
    poolStride = 0  # set to 0 to disable pooling
    input_channels = 5
    output_channels = 5
##############################################
# For Grav Intensity
##############################################
elif data_name == "GravInts":
    S = 64
    T = 5
    T_in = 5
    DATA_PATH = "../dataToSend/TrainingData/Intensity/"
    dN = 10
    mass = "ALL"
    # dt = 0.0204
    extras = "_intensity"

    TRAIN_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}{extras}.pt"
    TIME_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}{extras}.h5"
    N = 29
    if mass == "ALL":
        N = 902
    data_name = f"{data_name}{mass}_dN{dN}"
    input_channels = 5
    output_channels = 5
    modes = 8  # star Form 20
    width = 20  # star Form 100
    poolKernel = 0  # set to 0 to disable pooling
    poolStride = 0  # set to 0 to disable pooling
##############################################
# For MHD Star Formation Data
##############################################
elif data_name == "StarForm":
    S = 800
    T = 1
    T_in = 1
    DATA_PATH = "../dataToSend/TrainingData/"
    # DATA_PATH = "../dataToSend/FullDataTensor/"
    mu = "2"
    dN = 10
    sub = 1

    TRAIN_PATH = f"{DATA_PATH}density_mu{mu}_dN{dN}.pt"
    TIME_PATH = f"{DATA_PATH}time_mu{mu}_dN{dN}.h5"
    N = 874
    data_name = f"{data_name}_mu{mu}"
    input_channels = 1
    output_channels = 1
    modes = 20  # star Form 20
    width = 150  # star Form 100
    poolKernel = 0  # set to 0 to disable pooling
    poolStride = 0  # set to 0 to disable pooling

##############################################
# For CATS MHD Data
##############################################
elif data_name == "CATS":
    DATA_PATH = "../dataToSend/MHD_CATS/"
    TRAIN_PATH = f"{DATA_PATH}CATS_full.pt"
    TIME_PATH = f"{DATA_PATH}../CaltechData/ns_V1e-3_N5000_T50.h5"

    S = 256
    T = 1
    T_in = 1

    input_channels = 1
    output_channels = 1
    modes = 12
    width = 48
    N = 2970

if positional_encoding:
    input_channels += 2  # type: ignore

##############################################
# Data parameters
##############################################
split = [0.7, 0.2, 0.1]

# calculate the size of the output of the pooling layer from
# https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
if poolKernel > 0:
    S = math.floor((S - poolKernel) / poolStride + 1)

##############################################
# Neural network parameters
##############################################
NN = "RNN"  # "FNO2d", "MNO", "FNO3d" or "CNL2d"
factorization = None
rank = 0.001
norm = "group_norm"
g = (1, 1)
skip_type = "soft-gating"  # "identity", "linear" or "soft-gating"
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
##############################################
# Training parameters
##############################################
lr = 0.0001
weight_decay = 2e-4
epochs = 5
batch_size = 100
optimizer = "Adam"
scheduler_step = 10
cosine_step = 5
scheduler_gamma = 0.5
if NN == "FNO3d" or NN == "CNL2d":
    d = 3
else:
    d = 2
if loss_name == "LpLoss":
    loss_fn = LpLoss(d=d, p=2, reduce_dims=(0, 1), reductions="mean")
elif loss_name == "H1Loss":
    loss_fn = H1Loss(d=d, reduce_dims=(0, 1), reductions="mean")


file = "output.log"
log_interval = 100
