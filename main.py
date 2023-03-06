import os
from utils.utilsMain import *
import torch
import networks.networkUtils as myNet
import argparse
import logging
import time


def main(params, S=800):
    # grab the ending of the density file with .pt
    densityProfile = params.TRAIN_PATH.split("density")[-1].split(".")[0]
    S = S // params.sub
    # create output folder
    path = f"SF_{params.NN}_V100_ep{params.epochs}_m{params.modes}_w{params.width}_S{S}{densityProfile}"
    # check if path exists
    if not os.path.exists(f"results/{path}"):
        os.makedirs(f"results/{path}")
        os.makedirs(f"results/{path}/logs")
        os.makedirs(f"results/{path}/models")
        os.makedirs(f"results/{path}/plots")
        os.makedirs(f"results/{path}/data")
    print(f"Output folder for {path}")
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
    output_encoder.cuda()
    # log the model to debug
    logger.info(f"Neural network created in {time.time() - netTime} seconds")
    ################################################################
    # train neural network
    ################################################################
    logger.info("........Training neural network........")
    trainTime = time.time()
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
    testTime = time.time()
    # test neural network
    Trainer.evaluate(
        validLoader,
        validTime,
        output_encoder=output_encoder,
        input_encoder=input_encoder,
    )


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
