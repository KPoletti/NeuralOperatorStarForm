import argparse
import os
import sys
import torch
from dataclasses import dataclass
import pandas as pd
from neuralop.utils import UnitGaussianNormalizer
from neuralop.data.tensor_dataset import TensorDataset


class parserInputParameters(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str,
    ):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def dataSplit(params) -> tuple:
    tmp = pd.read_hdf(params.TIME_PATH, "table").shape[0]
    trainSize = int(tmp * params.split[0])
    testSize = int(tmp * params.split[1])
    validSize = tmp - trainSize - testSize
    return trainSize, testSize, validSize


def loadData(params: dataclass, isDensity: bool) -> tuple:
    """
    Load data from the path specified in the input file
    """
    if isDensity:
        filename = params.TRAIN_PATH
        fullData = torch.load(filename)[:, :: params.sub, :: params.sub]
    else:
        filename = params.TIME_PATH
        # TODO: Check formatting of time data to match the original data
        fullData = pd.read_hdf(filename, "table")
    # find the indices to split the data
    trainSize, testSize, validSize = dataSplit(params)
    # load data
    trainData = fullData[:trainSize]
    testData = fullData[trainSize : trainSize + testSize]
    validData = fullData[trainSize + testSize : trainSize + testSize + validSize]
    del fullData
    return trainData, testData, validData


def timestepSplit(
    dataSet: torch.tensor, dataInfo: pd.DataFrame, params: dataclass
) -> tuple:
    """
    Split the data into the time step to predict and the time step to use
    """
    dataSet_a = dataSet[..., : params.T_in]
    dataSet_u = dataSet[..., params.T_in : params.T_in + params.T]
    # split the data frame betwen the time step to predict and the time step to use
    dataInfo_a = dataInfo["t"]
    dataInfo_u = dataInfo["t+1"]
    return dataSet_a, dataSet_u, dataInfo_a, dataInfo_u


def prepareDataForTraining(params: dataclass, S: int) -> tuple:
    """ """
    # TODO: Write tests for this function

    trainSize, testsSize, validSize = dataSplit(params)
    # load data
    trainData, testsData, validData = loadData(params, isDensity=True)
    trainTime, testsTime, validTime = loadData(params, isDensity=False)
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

    if params.encoder:
        # normailze input data
        input_encoder = UnitGaussianNormalizer(trainData_a)
        trainData_a = input_encoder.encode(trainData_a)
        testsData_a = input_encoder.encode(testsData_a)
        validData_a = input_encoder.encode(validData_a)
        # normalize output data
        output_encoder = UnitGaussianNormalizer(trainData_u)
        trainData_u = output_encoder.encode(trainData_u)

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
    # Load the data into a tensor dataset
    trainDataset = TensorDataset(trainData_a, trainData_u)
    testsDataset = TensorDataset(testsData_a, testsData_u)
    validDataset = TensorDataset(validData_a, validData_u)
    # Create the data loader
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=params.batch_size, shuffle=True, drop_last=True
    )
    testLoader = torch.utils.data.DataLoader(
        testsDataset, batch_size=params.batch_size, shuffle=True, drop_last=True
    )
    validLoader = torch.utils.data.DataLoader(
        validDataset, batch_size=params.batch_size, shuffle=True, drop_last=True
    )
    return trainLoader, testLoader, validLoader, output_encoder
