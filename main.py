import os
from utils.utilsMain import *
import torch
import networks.networkUtils as myNet
import argparse
import logging
import time


def main(params, S=800):
    # find the way to split the data
    trainSize, testSize, validSize = dataSplit(params)
    # grab the ending of the density file with .pt
    densityProfile = params.TRAIN_PATH.split("density")[-1].split(".")[0]
    S = S // params.sub
    # create output folder
    path = f"SF_{params.NN}_V100_N{trainSize}_ep{params.epochs}_m{params.modes}_w{params.width}_S{S}{densityProfile}"
    # check if path exists
    if not os.path.exists(f"results/{path}"):
        os.makedirs(f"results/{path}")

    # SET UP LOGGING
    logging.basicConfig(filename=f"results/{path}/output.log", level=params.level)
    logging.info(f"Output folder for {path}")

    logging.debug(f"Full size: {trainSize + testSize + validSize}")
    logging.debug(f"Train size: {trainSize}")
    logging.debug(f"Test size: {testSize}")
    logging.debug(f"Valid size: {validSize}")
    ################################################################
    # load data
    ################################################################
    logging.info("Loading data")
    dataTime = time.time()
    trainLoader, testLoader, validLoader, output_encoder = prepareDataForTraining(
        params, S
    )
    logging.info(f"Data loaded in {time.time() - dataTime} seconds")
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################################################
    # create neural network
    ################################################################
    logging.info("Creating neural network")
    netTime = time.time()
    # create neural network
    model = myNet.initializeNetwork(params)

    # move to GPU
    model = model.to(device)
    # if params.NN == "MNO":
    # initailize dispative regularization
    # diss_loss = myNet.dissipativeRegularizer(
    #     params=params, model=model, x=vars, device=device
    # )
    logging.info(f"Neural network created in {time.time() - netTime} seconds")
    ################################################################
    # train neural network
    ################################################################
    logging.info("Training neural network")
    trainTime = time.time()
    # train neural network
    Trainer = myNet.Trainer(
        model=model,
        params=params,
        device=device,
    )
    Trainer.train(trainLoader, testLoader, output_encoder, regularizer=True)


if __name__ == "__main__":
    ################################################################
    # Parser
    ################################################################
    """
    Parse in command line arguments, these are defined in input.txt
    """
    os.chdir("/workspace/kpoletti/FNO/NeuralOperatorStarForm/")
    parser = argparse.ArgumentParser()
    ######### input parameters #########
    parser.add_argument("--input", type=str, default="input", help="input file")
    params = __import__(parser.parse_args().input)
    # params = imp.load_source("params", parser.parse_args().input)
    main(params, S=800)
