import argparse
import os
import sys
import torch
from dataclasses import dataclass
import pandas as pd

# from neuralop.data.transforms import PositionalEmbedding
from neuralop.utils import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
import input as param
import logging

logger = logging.getLogger(__name__)

# grab the ending of the density file with .pt
densityProfile = param.TRAIN_PATH.split("density")[-1].split(".")[0]

# create output folder
path = (
    f"SF_{param.NN}_V100_ep{param.epochs}"
    f"_m{param.modes}_w{param.width}_S{param.S}"
    f"{densityProfile}"
    f"_E{param.encoder}"
)

if not os.path.exists(f"results/{path}/logs"):
    os.makedirs(f"results/{path}/logs")
    os.makedirs(f"results/{path}/models")
    os.makedirs(f"results/{path}/plots")
    os.makedirs(f"results/{path}/data")
logger.setLevel(param.level)
fileHandler = logging.FileHandler(f"results/{path}/logs/util.log", mode="w")
formatter = logging.Formatter(
    "%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s"
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)


def dataSplit(params) -> tuple:
    """
    Finds the length of the time series and
    splits it into training, testing, and validation
    Input:
        params: input parameters from the input file
    Output:
        trainSize: int length of the training data
        testSize: int length of the testing data
        validSize: int length of the validation data
    """
    if params.log:
        tmp = pd.read_hdf(params.TIME_PATH, "table").shape[0]
    else:
        tmp = params.N
    trainSize = int(tmp * params.split[0])
    testSize = int(tmp * params.split[1])
    validSize = tmp - trainSize - testSize
    return trainSize, testSize, validSize


# TODO: Add average pooling function
def averagePooling(data: torch.tensor, params: dataclass) -> torch.tensor:
    """
    Average pool the data to reduce the size of the image
    Input:
        data: torch.tensor data to pool
        params: input parameters from the input file
    Output:
        data: torch.tensor data after pooling
    """
    logger.debug(f"Pooling data with kernel size {params.poolKernel}")
    # permute the data to pool across density
    data = data.permute(0, 3, 1, 2)
    data = torch.nn.functional.avg_pool2d(
        data, kernel_size=params.poolKernel, stride=params.poolStride
    )
    # permute the data back to the original shape
    data = data.permute(0, 2, 3, 1)
    return data


def loadData(params: dataclass, isDensity: bool) -> tuple:
    """
    Load data from the path specified in the input file
    Input:
        params: input parameters from the input file
        isDensity: if True load in the density data,
               if False load in the information about each time step
    Output:
        isDensity = True:

            trainData: torch.tensor training data,
            testingData: torch.tensor testing data,
            validData: torch.tensor validation data
                Each tensor has dimensions (N, S, S, 2)
                N is the number of timesteps
                S is the size of the image
                2 corresponds to the density at time t and t+1
        isDensity = False:
            trainData: pd.DataFrame training data,
            testingData: pd.DataFrame testing data,
            validData: pd.DataFrame validation data,
                each dataframe is split into df["t"] and df["t+1"]
                the dataframe can be indexed by dimension and time step
                Each df["t"] is a dataframe with the following columns:
                "time" time in Myr,
                "filename" name of the file corresponding to that data,
                "number" the number of timesteps,
    """
    if isDensity:
        filename = params.TRAIN_PATH
        fullData = torch.load(filename)
        if params.log:
            fullData = torch.log10(fullData).float()
        if params.poolKernel > 0:
            fullData = averagePooling(fullData, params)
    else:
        filename = params.TIME_PATH
        # load in data about each time step
        fullData = pd.read_hdf(filename, "table")
    # find the indices to split the data
    trainSize, testSize, validSize = dataSplit(params)
    # load data
    trainData = fullData[:trainSize]
    testData = fullData[trainSize : trainSize + testSize]
    validData = fullData[trainSize + testSize :]
    del fullData
    return trainData, testData, validData


def timestepSplit(
    dataSet: torch.tensor, dataInfo: pd.DataFrame, params: dataclass
) -> tuple:
    """
    Split the data into the time step to predict and the time step to use
    Input:
        dataSet: torch.tensor data to split
        dataInfo: pd.DataFrame information about each time step
        params: input parameters from the input file
    Output:
        dataSet_a: torch.tensor time step to use
        dataSet_u: torch.tensor time step to predict
        dataInfo_a: pd.DataFrame information about the time step to use
        dataInfo_u: pd.DataFrame information about the time step to predict
    """
    dataSet_a = dataSet[..., : params.T_in]
    dataSet_u = dataSet[..., params.T_in : params.T_in + params.T]
    # split the data frame betwen the time step to predict and the time step to use
    dataInfo_a = dataInfo["t"]
    dataInfo_u = dataInfo["t+1"]
    return dataSet_a, dataSet_u, dataInfo_a, dataInfo_u


class PositionalEmbedding:
    """Positional embedding for 2D data, from neuraloperator github"""

    def __init__(self, grid_boundaries, channel_dim):
        self.grid_boundaries = grid_boundaries
        self.channel_dim = channel_dim
        self._grid = None

    def grid(self, data):
        if self._grid is None:
            self._grid = addGrid(
                data, grid_boundaries=self.grid_boundaries, channel_dim=self.channel_dim
            )
        return self._grid

    def __call__(self, data):
        x, y = self.grid(data)

        return torch.cat((data, x, y), dim=-1)


def addGrid(
    input_tensor, grid_boundaries: list = [[-2.5, 2.5], [-2.5, 2.5]], channel_dim=0
):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional
    dimensions along the channels
    This is a modified version of the function in neuralop.data.transforms to allow work
    with 4D tensors
    """
    shape = list(input_tensor.shape)
    if len(shape) == 2:
        height, width = shape
    else:
        height, width, _ = shape

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing="ij")

    if len(shape) <= 3:
        grid_x = grid_x.repeat(1, 1).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(channel_dim)
    else:
        grid_x = grid_x.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)

    return grid_x, grid_y


def prepareDataForTraining(params: dataclass, S: int) -> tuple:
    """
    Loads in the data,
    Splits it into training, testing, and validation subsets
    Splits the data into the time step to predict and the time step to use
    Normalizes the data
    Reformats the data into the correct shape for the neural network
    Loads the data into pytorch dataLoaders
    Input:
        params: input parameters from the input file
        S: int size of the image
    Output:
        trainLoader: torch.utils.data.DataLoader training data
        testsLoader: torch.utils.data.DataLoader testing data
        validLoader: torch.utils.data.DataLoader validation data
        output_encoder: neuralop.utils.UnitGaussianNormalizer
    """
    # TODO: Write tests for this function

    trainSize, testsSize, validSize = dataSplit(params)
    logger.debug(f"Full size: {trainSize + testsSize + validSize}")
    logger.debug(f"Train size: {trainSize}")
    logger.debug(f"Test size: {testsSize}")
    logger.debug(f"Valid size: {validSize}")
    # load data
    trainData, testsData, validData = loadData(params, isDensity=True)
    trainTime, testsTime, validTime = loadData(params, isDensity=False)
    logger.debug(f"trainData shape: {trainData.shape}")
    # divide data into time step and timestep to predict

    # TODO: Figure out how to include the timesteps in the data
    trainData_a, trainData_u, trainTime_a, trainTime_u = timestepSplit(
        trainData, trainTime, params
    )
    testsData_a, testsData_u, testsTime_a, testsTime_u = timestepSplit(
        testsData, testsTime, params
    )
    validData_a, validData_u, validTime_a, validTime_u = timestepSplit(
        validData, validTime, params
    )

    # check if nn contains FNO
    if "FNO3d" in params.NN:
        trainData_a = trainData_a.reshape(trainSize, S, S, 1, params.T_in).repeat(
            [1, 1, 1, params.T, 1]
        )
        testsData_a = testsData_a.reshape(testsSize, S, S, 1, params.T_in).repeat(
            [1, 1, 1, params.T, 1]
        )
        validData_a = validData_a.reshape(validSize, S, S, 1, params.T_in).repeat(
            [1, 1, 1, params.T, 1]
        )
        trainData_a = trainData_a.permute(0, 3, 4, 1, 2)
        testsData_a = testsData_a.permute(0, 3, 4, 1, 2)
        validData_a = validData_a.permute(0, 3, 4, 1, 2)
    if ("FNO2d" in params.NN) or ("MNO" in params.NN):
        trainData_a = trainData_a.reshape(trainSize, S, S, params.T_in)
        testsData_a = testsData_a.reshape(testsSize, S, S, params.T_in)
        validData_a = validData_a.reshape(validSize, S, S, params.T_in)
        trainData_a = trainData_a.permute(0, 3, 1, 2)
        testsData_a = testsData_a.permute(0, 3, 1, 2)
        validData_a = validData_a.permute(0, 3, 1, 2)
        # permute u data
        trainData_u = trainData_u.permute(0, 3, 1, 2)
        testsData_u = testsData_u.permute(0, 3, 1, 2)
        validData_u = validData_u.permute(0, 3, 1, 2)

    if params.encoder:
        # normailze input data
        input_encoder = UnitGaussianNormalizer(trainData_a)
        trainData_a = input_encoder.encode(trainData_a)
        testsData_a = input_encoder.encode(testsData_a)
        validData_a = input_encoder.encode(validData_a)
        # normalize output data
        output_encoder = UnitGaussianNormalizer(trainData_u)
        trainData_u = output_encoder.encode(trainData_u)
        # testsData_u = output_encoder.encode(testsData_u)
        # validData_u = output_encoder.encode(validData_u)
    else:
        output_encoder = None
        input_encoder = None
    # reshape data

    # define the grid
    grid_bounds = [[-2.5, 2.5], [-2.5, 2.5]]
    # Load the data into a tensor dataset
    trainDataset = TensorDataset(
        trainData_a,
        trainData_u,
    )
    testsDataset = TensorDataset(
        testsData_a,
        testsData_u,
    )
    validDataset = TensorDataset(
        validData_a,
        validData_u,
    )
    # Create the data loader
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=params.batch_size, shuffle=True, drop_last=True
    )
    testLoader = torch.utils.data.DataLoader(
        testsDataset, batch_size=params.batch_size, shuffle=False, drop_last=True
    )
    validLoader = torch.utils.data.DataLoader(
        validDataset, batch_size=params.batch_size, shuffle=False, drop_last=True
    )
    return trainLoader, testLoader, validLoader, input_encoder, output_encoder
