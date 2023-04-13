import os
from utils.utilsMain import *
import torch
import numpy as np
import networks.networkUtils as myNet
import argparse
import logging
import time


def main(params):
    # grab the ending of the density file with .pt
    densityProfile = params.TRAIN_PATH.split("density")[-1].split(".")[0]
    N = (params.split[0] + params.split[1]) * params.N
    # create output folder
    path = (
        f"SF_{params.NN}_V100_ep{params.epochs}"
        f"_m{params.modes}_w{params.width}_S{params.S}"
        f"{densityProfile}"
        f"_E{params.encoder}"
        f"_N{N}"
    )
    # check if path exists
    if not os.path.exists(f"results/{path}"):
        os.makedirs(f"results/{path}")
        os.makedirs(f"results/{path}/logs")
        os.makedirs(f"results/{path}/models")
        os.makedirs(f"results/{path}/plots")
        os.makedirs(f"results/{path}/data")
    # SET UP LOGGING
    logger = logging.getLogger(__name__)
    logger.setLevel(params.level)
    fileHandler = logging.FileHandler(f"results/{path}/logs/output.log", mode="w")
    formatter = logging.Formatter(
        "%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(f"Output folder for {path}")

    ################################################################
    # load data
    ################################################################
    logger.info("........Loading data........")
    dataTime = time.time()
    trainTime, testsTime, validTime = loadData(params, isDensity=False)
    (
        trainLoader,
        testLoader,
        validLoader,
        input_encoder,
        output_encoder,
    ) = prepareDataForTraining(params, params.S)
    logger.info(f"Data loaded in {time.time() - dataTime} seconds")
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################################################
    # create neural network
    ################################################################
    logger.info("........Creating neural network........")
    netTime = time.time()
    # create neural network
    model = myNet.initializeNetwork(params)
    # move to GPU
    model = model.to(device)
    if params.encoder:
        output_encoder.cuda()
        input_encoder.cuda()
    # log the model to debug
    logger.info(f"Neural network created in {time.time() - netTime} seconds")
    ################################################################
    # train neural network
    ################################################################
    logger.info("........Training neural network........")
    # train neural network
    Trainer = myNet.Trainer(
        model=model,
        params=params,
        device=device,
    )
    Trainer.train(trainLoader, testLoader, output_encoder)
    ################################################################
    # test neural network
    ################################################################
    logger.info("........Testing neural network........")
    # test neural network
    Trainer.evaluate(
        validLoader,
        validTime,
        output_encoder=output_encoder,
        input_encoder=input_encoder,
        # savename="ValidationData",
    )

    ## print output folder
    print(f"Output folder for {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    ################################################################
    # Parser
    ################################################################
    """
    Parse in command line arguments, these are defined in input.txt
    """
    # os.chdir("/work/08770/kp32595/ls6/research/NeuralOperatorStarForm/")
    parser = argparse.ArgumentParser()
    print(f"Found {torch.cuda.device_count()} GPUs")
    ######### input parameters #########
    parser.add_argument("--input", type=str, default="input", help="input file")
    params = __import__(parser.parse_args().input)
    parser.add_argument("--N", type=int, default=params.N, help="Number of Data points")
    main(params)
