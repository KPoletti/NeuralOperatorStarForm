"""
This module contains utility functions for the Neural Operator StarForm project.
Functions include data splitting, average pooling, and log10 normalization.
"""

import logging
from dataclasses import dataclass

import pandas as pd
import torch
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.utils import UnitGaussianNormalizer
from src.meta_dataset import TensorDataset
from torch.utils.data.distributed import DistributedSampler

import wandb

# initialize logger
logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)


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
    if "mu" in params.data_name:
        tmp = pd.read_hdf(params.TIME_PATH, "table").shape[0]
    else:
        tmp = params.N
    train_size = int(tmp * params.split[0])
    test_size = int(tmp * params.split[1])
    valid_size = int(tmp * params.split[2]) - 1
    # confirm no overlap or zeros
    try:
        assert train_size + test_size + valid_size == tmp
    except AssertionError:
        logger.warning("Data split is not correct. Manually setting split.")
        valid_size = tmp - train_size - test_size

    return train_size, test_size, valid_size


def averagePooling(data: torch.Tensor, params: dataclass) -> torch.Tensor:
    """
    Average pool the data to reduce the size of the image
    Input:
        data: torch.tensor data to pool
        params: input parameters from the input file
    Output:
        data: torch.tensor data after pooling
    """
    if params.poolKernel <= 0:
        return data
    logger.debug("Pooling data with kernel size %d", params.poolKernel)
    # permute the data to pool across density
    if len(data.shape) == 4:
        data = data.permute(0, 3, 1, 2)
        data = torch.nn.functional.avg_pool2d(
            data, kernel_size=params.poolKernel, stride=params.poolStride
        )
        # permute the data back to the original shape
        data = data.permute(0, 2, 3, 1)
    elif len(data.shape) == 5:
        kernel = (1, params.poolKernel, params.poolKernel)
        stride = (1, params.poolStride, params.poolStride)
        data = data.permute(0, 3, 4, 1, 2)
        data = torch.nn.functional.avg_pool3d(data, kernel_size=kernel, stride=stride)
        # permute the data back to the original shape
        data = data.permute(0, 3, 4, 1, 2)
    return data


class Log10Normalizer:
    """
    A class for normalizing data using a log10 transformation.
    """

    def __init__(self, data):
        """
        Initializes the Log10Normalizer class.
        Input:
            data: the data to be normalized
        """
        super().__init__()
        self.data = data
        self.min = data.min()
        self.max = data.max()
        self.shape = data.shape

    def encode(self, data):
        """
        Encodes the data using a log10 transformation.
        Input:
            data: the data to be encoded
        Output:
            data: the encoded data
        """
        if data.shape[-2] == 3:
            data[..., 0, :] = data[..., 0, :].log10()
            return data
        return data.log10()

    def decode(self, data):
        """
        Decodes the data using a log10 transformation.
        Input:
            data: the data to be decoded
        Output:
            data: the decoded data
        """
        base = torch.tensor(10.0)
        if data.shape[-2] == 3:
            data[..., 0, :] = base.pow(data[..., 0, :])
            return base.pow(data)
        return base.pow(data)

    def cuda(self):
        """
        Moves the Log10Normalizer object to the GPU.
        """
        return self


def reduceToDensity(data: torch.tensor, params: dataclass) -> torch.tensor:
    """
    Reduces the data to only the density
    Input:
        data: torch.tensor data to reduce
        params: input parameters from the input file
    Output:
        data: torch.tensor data after reduction
    """
    # if "GravColl" in params.data_name and "FNO2d" in params.NN:
    if "FNO2d_dens" in params.NN or "MNO" in params.NN:
        return data[..., 0, :]
    elif "RNN2d" in params.NN:
        return data[..., 0:1, :]
    return data


def logData(data: torch.tensor, params: dataclass) -> torch.tensor:
    """
    Log the data
    Input:
        data: torch.tensor data to log
        params: input parameters from the input file
    Output:
        data: torch.tensor data after logging
    """

    if params.log:
        log_encoder = Log10Normalizer(data)
        data = log_encoder.encode(data)
    return data


def loadData(params: dataclass, is_data: bool) -> tuple:
    """
    Load data from the path specified in the input file
    Input:
        params: input parameters from the input file
        isData: if True load in the density data,
               if False load in the information about each time step
    Output:
        isData = True:

            trainData: torch.tensor training data,
            testingData: torch.tensor testing data,
            validData: torch.tensor validation data
        isData = False:
            trainData: pd.DataFrame training data,
            testingData: pd.DataFrame testing data,
            validData: pd.DataFrame validation data,
    """
    if is_data:
        filename = params.TRAIN_PATH
        full_data = torch.load(filename)
        full_data = full_data.float()
        # reduce to density
        full_data = reduceToDensity(full_data, params)
        # log the data
        full_data = logData(full_data, params)
        # average pool the data
        full_data = averagePooling(full_data, params)
        logger.debug(f"Full Data Shape: {full_data.shape}")
    else:
        filename = params.TIME_PATH
        # load in data about each time step
        full_data = pd.read_hdf(filename, "table")
    # find the indices to split the data
    # trainData, testData, validData = train_test_valid_split(params, fullData)
    # del fullData
    return full_data


def trainTestValidSplit(params: dataclass, data: torch.Tensor) -> tuple:
    """
    Break the data into training, testing, and validation sets
    Input:
        params: input parameters from the input file
        data: torch.tensor data to split
    Output:
        trainData: torch.tensor training data,
        testingData: torch.tensor testing data,
        validData: torch.tensor validation data
            Each tensor has dimensions (N, S, S, 2)
            N is the number of timesteps
            S is the size of the image
            2 corresponds to the density at time t and t+1
    """
    train_size, test_size, valid_size = dataSplit(params)
    # check if sizes exceed total size.
    if train_size + test_size + valid_size > data.shape[0]:
        print("WARNING: Given N is larger than the number of data points.")
        print("Defaulting to number of data point.")
        N = data.shape[0]
        train_size = int(N * params.split[0])
        test_size = int(N * params.split[1])
        valid_size = int(N * params.split[2]) - 1
    # load data
    train_data = data[:train_size]
    test_data = data[train_size : train_size + test_size]
    valid_data = data[-valid_size:]
    return train_data, test_data, valid_data


def timestepSplit(
    dataset: torch.Tensor, data_info: pd.DataFrame, params: dataclass
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
    dataset_a = dataset[..., : params.T_in]
    dataset_u = dataset[..., params.T_in : params.T_in + params.T]
    # split the data frame betwen the time step to predict and the time step to use
    try:
        data_info_a = data_info["t"]
        data_info_u = data_info["t+1"]
    except KeyError:
        logger.warning("No dataInfo['t'] found")
        # get all the keys in the dataframe
        keys = data_info.keys()
        keys = [*set([key[0] for key in keys])]
        keys.sort()

        # find the keys that correspond to the time step to use t+{dN-1}
        keys_a = [key for key in keys if int(key.split("+")[1]) <= params.T_in - 1]
        keys_u = [key for key in keys if int(key.split("+")[1]) > params.T_in - 1]
        # split the dataframe
        data_info_a = data_info[keys_a]
        data_info_u = data_info[keys_u]
    return dataset_a, dataset_u, data_info_a, data_info_u


def dfTolist(frame: pd.DataFrame) -> list:
    """
    Convert a dataframe to a list of tensors
    Input:
        df: pd.DataFrame to convert
    Output:
        list of tensors
    """
    # convert the dataframe to a list of tensors

    data = {}
    # find keys of the dataframe
    keys = frame.keys()
    for i in range(len(frame)):
        info = {}
        # find the mass of the star
        snap = frame.iloc[i][keys[0][0]]["file"]
        # split the file by mass
        info["mass"] = snap.split("_")[1]
        info["time"] = [frame.iloc[i][key[0]]["time"] for key in keys[::2]]
        # append info to the list
        data[i] = info
    return data


class MultiVariableNormalizer:
    """
    Normalize the data by the mean and standard deviation for velocity and density
    separately
    Input:
        data: torch.tensor data to normalize
    Output:
        data: torch.tensor normalized data
    """

    def __init__(self, data, verbose):
        super().__init__()
        self.data = data
        self.shape = torch.tensor(data.shape)
        logger.debug("Encoder Data Shape: %s", data.shape)
        self.index = int([i for i, x in enumerate(self.shape) if x == 3][0])
        self.vel_index = torch.tensor([1, 2])
        self.rho_index = torch.tensor([0])
        self.rho_encoder = UnitGaussianNormalizer(
            data.index_select(self.index, self.rho_index), verbose=verbose
        )
        self.vel_encoder = UnitGaussianNormalizer(
            data.index_select(self.index, self.vel_index), verbose=verbose
        )

    def encode(self, data):
        """
        Encode the data by normalizing the density and velocity separately
        Input:
            data: torch.tensor data to encode
        Output:
            data: torch.tensor encoded data
        """
        rho = self.rho_encoder.encode(data.index_select(self.index, self.rho_index))
        vel = self.vel_encoder.encode(data.index_select(self.index, self.vel_index))
        # data[..., 0] = rho
        # data[..., 1:] = vel
        return torch.cat((rho, vel), dim=self.index)

    def decode(self, data):
        """
        Decode the data by denormalizing the density and velocity separately
        Input:
            data: torch.tensor data to decode
        Output:
            data: torch.tensor decoded data
        """
        rho = self.rho_encoder.decode(data.index_select(self.index, self.rho_index))
        vel = self.vel_encoder.decode(data.index_select(self.index, self.vel_index))
        # data[..., 0] = rho
        # data[..., 1:] = vel
        return torch.cat((rho, vel), dim=self.index)

    def cuda(self):
        """
        Move the model to the GPU
        """
        self.rho_encoder.cuda()
        self.vel_encoder.cuda()
        self.rho_index = self.rho_index.cuda()
        self.vel_index = self.vel_index.cuda()
        return self

    def cpu(self):
        """
        Move the model to the CPU
        """
        self.rho_encoder.cpu()
        self.vel_encoder.cpu()
        self.rho_index = self.rho_index.cpu()
        self.vel_index = self.vel_index.cpu()
        return self

    def to(self, device):
        """
        Move the model to the specified device
        """
        self.rho_encoder.to(device)
        self.vel_encoder.to(device)
        self.rho_index = self.rho_index.to(device)
        self.vel_index = self.vel_index.to(device)
        return self


def permuteFNO3d(data: torch.tensor) -> torch.tensor:
    """Block permutes the data to the correct shape for the FNO3d network

    Args:
        data (torch.tensor): data to permute

    Returns:
        torch.tensor: permuted data
    """
    data = data.permute(0, 3, 1, 2, 4)
    return data


def permuteFNO2d(
    data: torch.tensor, params: dataclass, size: int, gridsize: int
) -> torch.tensor:
    """Block permutes the data to the correct shape for the FNO2d network

    Args:
        data (torch.tensor): data to permute
        params (dataclass): input parameters from the input file
        size (int): number of samples
        gridsize (int): gridsize of the data

    Returns:
        torch.tensor: permuted data
    """
    # data = data.reshape(size, gridsize, gridsize, params.T_in)
    data = data.squeeze()
    print("WARNING FNO2D NO LONGER JUST DENSITY")
    data = data.permute(0, 3, 1, 2)
    return data


def permuteCNL2d(data: torch.tensor) -> torch.tensor:
    """Block permutes the data to the correct shape for the CNL2d network"""
    return data.permute(0, 4, 1, 2, 3)


def permute(data, params, size, gridsize) -> torch.tensor:
    """
    permute the data to the correct shape depending on the nn and the input file
    Input:
        data: torch.tensor data to permute
        params: input parameters from the input file
    Output:
        data: torch.tensor permuted data
    """
    if ("FNO3d" in params.NN) or ("RNN" in params.NN):
        return permuteFNO3d(data)
        # return permute_CNL2d(data)

    elif ("FNO2d" in params.NN) or ("MNO" in params.NN):
        return permuteFNO2d(data, params, size, gridsize)

    elif "CNL2d" in params.NN:
        return permuteCNL2d(data)
    else:
        raise ValueError(f"Unknown Neural Network: {params.NN}")


def initializeEncoder(data_a, data_u, params: dataclass, verbosity=False) -> tuple:
    """
    Initialize the input and output encoders for the given data
    Input:
        data_a: torch.tensor data for the input encoder
        data_u: torch.tensor data for the output encoder
        params: input parameters from the input file
        verbosity: boolean flag for verbose output
    Output:
        input_encoder: initialized input encoder
        output_encoder: initialized output encoder
    """
    if "FNO2d" in params.NN or "MNO" in params.NN or "Ints" in params.DATA_PATH:
        input_encoder = UnitGaussianNormalizer(data_a, verbose=verbosity)
        output_encoder = UnitGaussianNormalizer(data_u, verbose=verbosity)
    elif "GravColl" in params.data_name or "Turb" in params.data_name:
        input_encoder = MultiVariableNormalizer(data_a, verbose=verbosity)
        output_encoder = MultiVariableNormalizer(data_u, verbose=verbosity)
    else:
        input_encoder = UnitGaussianNormalizer(data_a, verbose=verbosity)
        output_encoder = UnitGaussianNormalizer(data_u, verbose=verbosity)
    input_encoder = UnitGaussianNormalizer(data_a, verbose=verbosity)
    output_encoder = UnitGaussianNormalizer(data_u, verbose=verbosity)
    return input_encoder, output_encoder


def prepareDataForTraining(params: dataclass, S: int) -> tuple:
    """
    1. Loads in the data,
    2. Splits it into training, testing, and validation subsets
    3. Splits the data into the time step to predict and the time step to use
    4. Normalizes the data
    5. Reformats the data into the correct shape for the neural network
    6. Loads the data into pytorch dataLoaders
    Input:
        params: input parameters from the input file
        S: int size of the image
    Output:
        trainLoader: torch.utils.data.DataLoader training dataSet
        testsLoader: torch.utils.data.DataLoader testing dataSet
        validLoader: torch.utils.data.DataLoader validation dataSet
        input_encoder: neuralop.utils.UnitGaussianNormalizer
        output_encoder: neuralop.utils.UnitGaussianNormalizer
    """
    # TODO: Write tests for this function

    train_size, tests_size, valid_size = dataSplit(params)
    logger.debug("Full size: %s", train_size + tests_size + valid_size)

    logger.debug("Train size: %s", train_size)
    logger.debug("Test size: %s", tests_size)
    logger.debug("Valid size: %s", valid_size)

    wandb.config["Train data"] = train_size
    wandb.config["Test data"] = tests_size
    wandb.config["Valid data"] = valid_size
    wandb.config["Total data"] = train_size + tests_size + valid_size

    # load data
    full_data = loadData(params, is_data=True)
    time_data = loadData(params, is_data=False)
    # divide data into time step and timestep to predict
    full_data_a, full_data_u, time_data_a, time_data_u = timestepSplit(
        full_data, time_data, params
    )

    # permute data to correct shape
    full_data_a = permute(full_data_a, params, full_data_a.shape[0], S)
    full_data_u = permute(full_data_u, params, full_data_u.shape[0], S)

    # split data into training, testing, and validation subsets
    train_data_a, tests_data_a, valid_data_a = trainTestValidSplit(params, full_data_a)
    train_data_u, tests_data_u, valid_data_u = trainTestValidSplit(params, full_data_u)

    train_time_a, tests_time_a, valid_time_a = trainTestValidSplit(params, time_data_a)
    train_time_u, tests_time_u, valid_time_u = trainTestValidSplit(params, time_data_u)

    # TODO: Figure out how to include the timesteps in the
    if params.level == "DEBUG":
        print(f"Train Data Shape: {train_data_a.shape}")
        print(f"Test Data Shape: {tests_data_a.shape}")
        print(f"Valid Data Shape: {valid_data_a.shape}")

    if params.encoder:
        # initialize the encoder
        input_encoder, output_encoder = initializeEncoder(
            train_data_a, train_data_u, params
        )
        # normalize input and ouput data
        train_data_a = input_encoder.encode(train_data_a)
        tests_data_a = input_encoder.encode(tests_data_a)
        valid_data_a = input_encoder.encode(valid_data_a)

        train_data_u = output_encoder.encode(train_data_u)
    else:
        output_encoder = None
        input_encoder = None

    # define the grid
    logger.debug("Train Data A Shape: %s", train_data_a.shape)
    logger.debug("Train Data U Shape: %s", train_data_u.shape)
    logger.debug("Test Data A Shape: %s", tests_data_a.shape)
    logger.debug("Test Data U Shape: %s", tests_data_u.shape)
    logger.debug("Valid Data A Shape: %s", valid_data_a.shape)
    logger.debug("Valid Data U Shape: %s", valid_data_u.shape)

    # Assert that the data is the correct shape
    try:
        assert train_data_a.shape == train_data_u.shape
        assert tests_data_a.shape == tests_data_u.shape
        assert valid_data_a.shape == valid_data_u.shape
    except AssertionError:
        logger.error("Data shapes do not match. Manually fixing.")

    # Load the data into a tensor dataset
    train_dataset = TensorDataset(
        train_data_a,
        train_data_u,
        dfTolist(train_time_a),
        dfTolist(train_time_u),
        transform_x=(
            PositionalEmbedding2D(params.grid_boundaries, 0)
            if params.positional_encoding
            else None
        ),
    )
    tests_dataset = TensorDataset(
        tests_data_a,
        tests_data_u,
        dfTolist(tests_time_a),
        dfTolist(tests_time_u),
        transform_x=(
            PositionalEmbedding2D(params.grid_boundaries, 0)
            if params.positional_encoding
            else None
        ),
    )
    valid_dataset = TensorDataset(
        valid_data_a,
        valid_data_u,
        dfTolist(valid_time_a),
        dfTolist(valid_time_u),
        transform_x=(
            PositionalEmbedding2D(params.grid_boundaries, 0)
            if params.positional_encoding
            else None
        ),
    )
    shuffle_type = True
    train_sampler = None
    test_sampler = None
    if params.use_ddp:
        shuffle_type = False
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(tests_dataset)
        tests_size //= torch.cuda.device_count()
    # Create the data loader
    # TODO: Fix the batch size for the tests and valid loaders
    # On gravcoll, for (15, 10, 200, 200 ,3) the test loss will grow if validSize =0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=shuffle_type,
        sampler=train_sampler,
        drop_last=True,
    )
    if tests_size >= params.batch_size * 2:
        test_loader = torch.utils.data.DataLoader(
            tests_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=True,
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            tests_dataset,
            batch_size=1,
            shuffle=False,
            sampler=test_sampler,
            drop_last=True,
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, drop_last=True
    )
    return train_loader, test_loader, valid_loader, input_encoder, output_encoder
