import os
from utils.utilsMain import *
import torch
import networks
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
