import torch
from dataclasses import dataclass
import pandas as pd
import wandb
from neuralop.utils import UnitGaussianNormalizer
from src.meta_dataset import TensorDataset
import logging
import numpy as np

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
    trainSize = int(tmp * params.split[0])
    testSize = int(tmp * params.split[1])
    validSize = int(tmp * params.split[2]) - 1
    # confirm no overlap or zeros
    try:
        assert trainSize + testSize + validSize == tmp
    except AssertionError:
        logger.warning("Data split is not correct. Manually setting split.")
        validSize = tmp - trainSize - testSize

    return trainSize, testSize, validSize


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
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.min = data.min()
        self.max = data.max()
        self.shape = data.shape

    def encode(self, data):
        if data.shape[-2] == 3:
            data[..., 0, :] = data[..., 0, :].log10()
            return data
        else:
            return data.log10()

    def decode(self, data):
        base = torch.tensor(10.0)
        if data.shape[-2] == 3:
            data[..., 0, :] = base.pow(data[..., 0, :])
            return base.pow(data)
        else:
            return base.pow(data)

    def cuda(self):
        return self


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
        fullData = fullData.float()
        if params.log:
            log_encoder = Log10Normalizer(fullData)
            fullData = log_encoder.encode(fullData)
        if params.poolKernel > 0:
            fullData = averagePooling(fullData, params)
        logger.debug(f"Full Data Shape: {fullData.shape}")
    else:
        filename = params.TIME_PATH
        # load in data about each time step
        fullData = pd.read_hdf(filename, "table")
    # find the indices to split the data
    trainSize, testSize, validSize = dataSplit(params)
    # load data
    trainData = fullData[:trainSize]
    testData = fullData[trainSize : trainSize + testSize]
    validData = fullData[-validSize:]
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
    try:
        dataInfo_a = dataInfo["t"]
        dataInfo_u = dataInfo["t+1"]
    except KeyError:
        logger.warning("No dataInfo['t'] found")
        # get all the keys in the dataframe
        keys = dataInfo.keys()
        keys = [*set([key[0] for key in keys])]
        keys.sort()

        # find the keys that correspond to the time step to use t+{dN-1}
        keys_a = [key for key in keys if int(key.split("+")[1]) <= params.T_in - 1]
        keys_u = [key for key in keys if int(key.split("+")[1]) >= params.T_in - 1]
        # split the dataframe
        dataInfo_a = dataInfo[keys_a]
        dataInfo_u = dataInfo[keys_u]
    return dataSet_a, dataSet_u, dataInfo_a, dataInfo_u


def dfTolist(df: pd.DataFrame) -> list:
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
    keys = df.keys()
    for i in range(len(df)):
        info = {}
        # find the mass of the star
        snap = df.iloc[i][keys[0][0]]["file"]
        # split the file by mass
        info["mass"] = snap.split("M")[-1].split("_")[0]
        info["time"] = [df.iloc[i][key[0]]["time"] for key in keys[::2]]
        # append info to the list
        data[i] = info
    return data


class multiVarible_normalizer:
    """
    Normalize the data by the mean and standard deviation for velocity and density separately
    Input:
        data: torch.tensor data to normalize
    Output:
        data: torch.tensor normalized data
    """

    def __init__(self, data, verbose):
        super().__init__()
        self.data = data
        self.rho = data
        self.rho_encoder = UnitGaussianNormalizer(data[..., 0], verbose=verbose)
        self.vel_encoder = UnitGaussianNormalizer(data[..., 1:], verbose=verbose)

    def encode(self, data):
        rho = self.rho_encoder.encode(data[..., 0])
        vel = self.vel_encoder.encode(data[..., 1:])
        data[..., 0] = rho
        data[..., 1:] = vel
        return data

    def decode(self, data):
        rho = self.rho_encoder.decode(data[..., 0])
        vel = self.vel_encoder.decode(data[..., 1:])
        data[..., 0] = rho
        data[..., 1:] = vel
        return data

    def cuda(self):
        self.rho_encoder.cuda()
        self.vel_encoder.cuda()
        return self

    def cpu(self):
        self.rho_encoder.cpu()
        self.vel_encoder.cpu()
        return self

    def to(self, device):
        self.rho_encoder.to(device)
        self.vel_encoder.to(device)
        return self


def permute_FNO3d(
    data: torch.tensor, params: dataclass, size: int, gridsize: int
) -> torch.tensor:
    data = data.reshape(size, gridsize, gridsize, 1, params.T_in)
    data = data.repeat([1, 1, 1, params.T, 1])
    data = data.permute(0, 4, 3, 1, 2)
    return data


def permute_FNO2d(
    data: torch.tensor, params: dataclass, size: int, gridsize: int
) -> torch.tensor:
    data = data.reshape(size, gridsize, gridsize, params.T_in)
    data = data.permute(0, 3, 1, 2)
    return data


def permute_CNL2d(data: torch.tensor) -> torch.tensor:
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
    if "FNO3d" in params.NN:
        return permute_FNO3d(data, params, size, gridsize)

    elif ("FNO2d" in params.NN) or ("MNO" in params.NN):
        return permute_FNO2d(data, params, size, gridsize)

    elif "CNL2d" in params.NN:
        return permute_CNL2d(data)
    else:
        raise ValueError(f"Unknown Neural Network: {params.NN}")


def initializeEncoder(data_a, data_u, params: dataclass, verbosity=False) -> tuple:
    if "GravColl" in params.data_name:
        input_encoder = multiVarible_normalizer(data_a, verbose=verbosity)
        output_encoder = multiVarible_normalizer(data_u, verbose=verbosity)
    else:
        input_encoder = UnitGaussianNormalizer(data_a, verbose=verbosity)
        output_encoder = UnitGaussianNormalizer(data_u, verbose=verbosity)
    return input_encoder, output_encoder


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
        trainLoader: torch.utils.data.DataLoader training dataSet
        testsLoader: torch.utils.data.DataLoader testing dataSet
        validLoader: torch.utils.data.DataLoader validation dataSet
        input_encoder: neuralop.utils.UnitGaussianNormalizer
        output_encoder: neuralop.utils.UnitGaussianNormalizer
    """
    # TODO: Write tests for this function

    trainSize, testsSize, validSize = dataSplit(params)
    logger.debug(f"Full size: {trainSize + testsSize + validSize}")
    logger.debug(f"Train size: {trainSize}")
    logger.debug(f"Test size: {testsSize}")
    logger.debug(f"Valid size: {validSize}")
    wandb.config["Train data"] = trainSize
    wandb.config["Test data"] = testsSize
    wandb.config["Valid data"] = validSize
    wandb.config["Total data"] = trainSize + testsSize + validSize

    # load data
    trainData, testsData, validData = loadData(params, isDensity=True)
    trainTime, testsTime, validTime = loadData(params, isDensity=False)
    logger.debug(f"trainData shape: {trainData.shape}")

    # divide data into time step and timestep to predict
    trainData_a, trainData_u, trainTime_a, trainTime_u = timestepSplit(
        trainData, trainTime, params
    )
    testsData_a, testsData_u, testsTime_a, testsTime_u = timestepSplit(
        testsData, testsTime, params
    )
    validData_a, validData_u, validTime_a, validTime_u = timestepSplit(
        validData, validTime, params
    )
    # TODO: Figure out how to include the timesteps in the data
    if params.level == "DEBUG":
        print(f"Train Data Shape: {trainData_a.shape}")
        print(f"Test Data Shape: {testsData_a.shape}")
        print(f"Valid Data Shape: {validData_a.shape}")

    # permute data
    trainData_a = permute(trainData_a, params, trainSize, S)
    testsData_a = permute(testsData_a, params, testsSize, S)
    validData_a = permute(validData_a, params, validSize, S)

    trainData_u = permute(trainData_u, params, trainSize, S)
    testsData_u = permute(testsData_u, params, testsSize, S)
    validData_u = permute(validData_u, params, validSize, S)

    if params.encoder:
        # initialize the encoder
        input_encoder, output_encoder = initializeEncoder(
            trainData_a, trainData_u, params
        )
        # normalize input and ouput data
        trainData_a = input_encoder.encode(trainData_a)
        testsData_a = input_encoder.encode(testsData_a)
        validData_a = input_encoder.encode(validData_a)

        trainData_u = output_encoder.encode(trainData_u)
    else:
        output_encoder = None
        input_encoder = None

    # define the grid
    grid_bounds = [[-2.5, 2.5], [-2.5, 2.5]]
    logger.debug(f"Train Data A Shape: {trainData_a.shape}")
    logger.debug(f"Train Data U Shape: {trainData_u.shape}")
    logger.debug(f"Test Data A Shape: {testsData_a.shape}")
    logger.debug(f"Test Data U Shape: {testsData_u.shape}")
    logger.debug(f"Valid Data A Shape: {validData_a.shape}")
    logger.debug(f"Valid Data U Shape: {validData_u.shape}")

    # Assert that the data is the correct shape
    try:
        assert trainData_a.shape == trainData_u.shape
        assert testsData_a.shape == testsData_u.shape
        assert validData_a.shape == validData_u.shape
    except AssertionError:
        logger.error("Data shapes do not match. Manually fixing.")

    # Load the data into a tensor dataset
    trainDataset = TensorDataset(
        trainData_a,
        trainData_u,
        dfTolist(trainTime_a),
        dfTolist(trainTime_u),
    )
    testsDataset = TensorDataset(
        testsData_a,
        testsData_u,
        dfTolist(testsTime_a),
        dfTolist(testsTime_u),
    )
    validDataset = TensorDataset(
        validData_a,
        validData_u,
        dfTolist(validTime_a),
        dfTolist(validTime_u),
    )
    # Create the data loader

    # TODO: Fix the batch size for the tests and valid loaders
    # On gravcoll, for (15, 10, 200, 200 ,3) the test loss will grow if validSize =0
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=params.batch_size, shuffle=True, drop_last=True
    )
    if testsSize >= params.batch_size * 2:
        testLoader = torch.utils.data.DataLoader(
            testsDataset, batch_size=params.batch_size, shuffle=False, drop_last=True
        )
    else:
        testLoader = torch.utils.data.DataLoader(
            testsDataset, batch_size=1, shuffle=False, drop_last=True
        )
    validLoader = torch.utils.data.DataLoader(
        validDataset, batch_size=1, shuffle=False, drop_last=True
    )
    return trainLoader, testLoader, validLoader, input_encoder, output_encoder


def deprecatedPrepareDataForTraining(params: dataclass, S: int) -> tuple:
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
            trainLoader: torch.utils.data.DataLoader training dataS
    Sdata
            output_encoder: neuralop.utils.UnitGaussianNormalizer
    """
    # TODO: Write tests for this function

    trainSize, testsSize, validSize = dataSplit(params)
    logger.debug(f"Full size: {trainSize + testsSize + validSize}")
    logger.debug(f"Train size: {trainSize}")
    logger.debug(f"Test size: {testsSize}")
    logger.debug(f"Valid size: {validSize}")
    wandb.config["Train data"] = trainSize
    wandb.config["Test data"] = testsSize
    wandb.config["Valid data"] = validSize
    wandb.config["Total data"] = trainSize + testsSize + validSize

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
    if params.level == "DEBUG":
        print(f"Train Data Shape: {trainData_a.shape}")
        print(f"Test Data Shape: {testsData_a.shape}")
        print(f"Valid Data Shape: {validData_a.shape}")

    # check if nn contains FNO
    if "FNO3d" in params.NN:
        trainData_a_deprecated = trainData_a.reshape(
            trainSize, S, S, 1, params.T_in
        ).repeat([1, 1, 1, params.T, 1])
        testsData_a_deprecated = testsData_a.reshape(
            testsSize, S, S, 1, params.T_in
        ).repeat([1, 1, 1, params.T, 1])
        validData_a_deprecated = validData_a.reshape(
            validSize, S, S, 1, params.T_in
        ).repeat([1, 1, 1, params.T, 1])
        trainData_a_deprecated = trainData_a_deprecated.permute(0, 3, 4, 1, 2)
        testsData_a_deprecated = testsData_a_deprecated.permute(0, 3, 4, 1, 2)
        validData_a_deprecated = validData_a_deprecated.permute(0, 3, 4, 1, 2)
        # permute u data
        trainData_u_deprecated = trainData_u.reshape(
            trainSize, S, S, 1, params.T_in
        ).repeat([1, 1, 1, params.T, 1])
        testsData_u_deprecated = testsData_u.reshape(
            testsSize, S, S, 1, params.T_in
        ).repeat([1, 1, 1, params.T, 1])
        validData_u_deprecated = validData_u.reshape(
            validSize, S, S, 1, params.T_in
        ).repeat([1, 1, 1, params.T, 1])
        trainData_u_deprecated = trainData_u_deprecated.permute(0, 3, 4, 1, 2)
        testsData_u_deprecated = testsData_u_deprecated.permute(0, 3, 4, 1, 2)
        validData_u_deprecated = validData_u_deprecated.permute(0, 3, 4, 1, 2)

    elif ("FNO2d" in params.NN) or ("MNO" in params.NN):
        trainData_a_deprecated = trainData_a.reshape(trainSize, S, S, params.T_in)
        testsData_a_deprecated = testsData_a.reshape(testsSize, S, S, params.T_in)
        validData_a_deprecated = validData_a.reshape(validSize, S, S, params.T_in)
        trainData_a_deprecated = trainData_a.permute(0, 3, 1, 2)
        testsData_a_deprecated = testsData_a.permute(0, 3, 1, 2)
        validData_a_deprecated = validData_a.permute(0, 3, 1, 2)
        # permute u data
        trainData_u_deprecated = trainData_u.permute(0, 3, 1, 2)
        testsData_u_deprecated = testsData_u.permute(0, 3, 1, 2)
        validData_u_deprecated = validData_u.permute(0, 3, 1, 2)

    elif "CNL2d" in params.NN:
        num_dims = trainData.shape[-2]
        # trainData_a_deprecated = trainData_a_deprecated.reshape(trainSize, params.T_in, S, S, num_dims)
        # testsData_a_deprecated = testsData_a_deprecated.reshape(testsSize, params.T_in, S, S, num_dims)
        # validData_a = validData_a.reshape(validSize, params.T_in, S, S, num_dims)
        x_a = trainData_a.permute(0, 4, 1, 2, 3)
        testsData_a_deprecated = testsData_a.permute(0, 4, 1, 2, 3)
        validData_a_deprecated = validData_a.permute(0, 4, 1, 2, 3)

        # select rand ind of Train data
        ind_t = np.random.randint(0, trainData_a.shape[-1])
        ind_pt = np.random.randint(0, trainData_a.shape[0])

        # assert that the data is correctly permuted
        torch.testing.assert_close(
            x_a[ind_pt, ind_t, :, :, :], trainData_a[ind_pt, :, :, :, ind_t]
        )
        trainData_a_deprecated = x_a

        # permute u data
        trainData_u_deprecated = trainData_u.permute(0, 4, 1, 2, 3)
        testsData_u_deprecated = testsData_u.permute(0, 4, 1, 2, 3)
        validData_u_deprecated = validData_u.permute(0, 4, 1, 2, 3)
