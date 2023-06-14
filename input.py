import torch.nn as nn
from src.dissipative_utils import (
    sample_uniform_spherical_shell,
    linear_scale_dissipative_target,
)
from neuralop.training.losses import LpLoss, H1Loss
import math

data_name = "GravColl"  # NS-Caltech, StarForm, GravColl or CATS


# Pooling parameters
poolKernel = 4  # set to 0 to disable pooling
poolStride = 4  # set to 0 to disable pooling
#############################################
# FOR NS Cal-tech Data
#############################################
if data_name == "NS-Caltech":
    DATA_PATH = "../dataToSend/CaltechData/"
    TRAIN_PATH = f"{DATA_PATH}ns_V1e-3_N5000_T50.pt"
    TIME_PATH = f"{DATA_PATH}ns_V1e-3_N5000_T50.h5"
    log = False
    S = 64
    N = 5000
    input_channels = 10
    output_channels = 10
    modes = 12  # star Form 20
    width = 24  # star Form 100
    T = 10
    T_in = 10
    poolKernel = 0  # set to 0 to disable pooling
    poolStride = 0  # set to 0 to disable pooling

##############################################
# For Grav Collapse
##############################################
elif data_name == "GravColl":
    S = 200
    T = 5
    T_in = 5
    DATA_PATH = "../dataToSend/TrainingData/Simulations/"
    dN = 10
    # two options for mass: "_M1.2" or "_M1"
    # mass = "ALL"
    # extras = ""
    # if mass == "_M1.2":
    #     dt = 0.019
    # elif mass == "_M1":
    #     dt = 0.02
    # elif mass == "1.2" or mass == "ALL":
    #     dt = 0.0186
    #     extras = "_multiData"
    # else:
    #     dt = 0.02
    # sub = 1
    # Grav_M1.00_dN10_dt0.0204_multiData
    mass = "ALL"
    # dt = 0.0204
    extras = "_multiData"
    # TRAIN_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}_dt{dt}{extras}.pt"
    # TIME_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}_dt{dt}{extras}.h5"

    TRAIN_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}{extras}_pooled.pt"
    TIME_PATH = f"{DATA_PATH}Grav_M{mass}_dN{dN}{extras}.h5"
    log = True
    N = 14
    if mass == "ALL":
        N = 902
    data_name = f"{data_name}{mass}_dN{dN}"
    input_channels = 5
    output_channels = 5
    modes = 10  # star Form 20
    width = 24  # star Form 100
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
    log = True
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

    log = True

    S = 256
    T = 1
    T_in = 1

    input_channels = 1
    output_channels = 1
    modes = 12
    width = 48
    N = 2970


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
NN = "CNL2d"  # "FNO2d", "MNO", "FNO" or "CNL2d"

encoder = False
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
epochs = 100
lr = 0.0001
scheduler_step = 50
scheduler_gamma = 0.5
batch_size = 10
optimizer = "Adam"
loss_name = "LpLoss"
if NN == "FNO3d" or NN == "CNL2d":
    d = 3
else:
    d = 2
if loss_name == "LpLoss":
    loss_fn = LpLoss(d=d, p=2, reduce_dims=(0, 1))
elif loss_name == "H1Loss":
    loss_fn = H1Loss(d=d, reduce_dims=(0, 1))


level = "DEBUG"
file = "output.log"
log_interval = 100

# Option to Save NN
saveNeuralNetwork = False

# Option to create plots
doPlot = True
