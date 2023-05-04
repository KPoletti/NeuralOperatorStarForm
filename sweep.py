"""
File for hyperparameter tuning with wandb sweep
"""

import os
from utils.utilsMain import *
import torch
import numpy as np
import networks.networkUtils as myNet
import argparse
import logging
import time
import wandb
import json


def convertParamsToDict(params):
    """
    Converts the params to a dictionary for wandb
    """
    paramsDict = {}
    unneeded = [
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
        "nn",
        "sample_uniform_spherical_shell",
        "linear_scale_dissipative_target",
        "LpLoss",
        "H1Loss",
        "math",
    ]
    for key, value in vars(params).items():
        # skip the levels that are not needed
        if key in unneeded:
            continue
        paramsDict[key] = value
    return paramsDict


def convertParamsToJSON(params):
    """
    Converts the params to a JSON for wandb
    """
    paramsDict = convertParamsToDict(params)
    paramsJSON = json.dumps(
        paramsDict, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )
    # convert back to dictionary
    paramsJSON = json.loads(paramsJSON)
    return paramsJSON


def set_Loss(params):
    from neuralop.training.losses import LpLoss, H1Loss

    if params.loss_name == "LpLoss":
        return LpLoss(d=2, p=2, reduce_dims=(0, 1))
    elif params.loss_name == "H1Loss":
        return H1Loss(d=1, reduce_dims=(0, 1))
    else:
        raise ValueError(f"Loss {params.loss_name} not implemented")


def main(config=None):
    params = __import__(parser.parse_args().input)

    with wandb.init(project=f"{params.data_name}_sweep", config=config):
        config = wandb.config
        # reset values based on sweep
        # TODO: make this more general
        for attr in config.keys():
            setattr(params, attr, config[attr])
        N = int((params.split[0] + params.split[1]) * params.N)
        # create output folder
        path = (
            f"SF_{params.NN}_{params.data_name}_ep{params.epochs}"
            f"_m{params.modes}_w{params.width}_S{params.S}"
            f"_E{params.encoder}"
            f"_N{N}"
        )
        # define the loss function
        params.loss_fn = set_Loss(params)
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
        params.batch_size = torch.cuda.device_count() * params.batch_size
        # move to GPU
        # model = torch.nn.DataParallel(model)
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
        )
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
    # change the directory to main.py's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    print(f"Found {torch.cuda.device_count()} GPUs")
    ######### input parameters #########
    parser.add_argument("--input", type=str, default="input", help="input file")
    params = __import__(parser.parse_args().input)
    parser.add_argument("--N", type=int, default=params.N, help="Number of Data points")

    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary
    paramsDict = convertParamsToJSON(params)
    # define the sweep_config
    sweep_config = {
        "method": "random",
        "metric": {"name": "Test-Loss", "goal": "minimize"},
    }
    # create the sweep parameters
    sweep_params = {
        "modes": {"distribution": "int_uniform", "min": 1, "max": 48},
        "width": {
            "distribution": "q_log_uniform_values",
            "q": 8,
            "min": 32,
            "max": 256,
        },
        "encoder": {"values": [True, False]},
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "scheduler_step": {"values": [10, 25, 50, 75]},
        # "loss_name": {"values": ["LpLoss", "H1Loss"]},
    }
    # create a list of the sweep parameters
    sweep_params_list = list(sweep_params.keys())
    # loop over paramsDict and to sweep params if they are not in sweep_params
    for key, value in paramsDict.items():
        if key not in sweep_params.keys():
            sweep_params[key] = {"value": value}
    # add sweep parameters to sweep_config
    sweep_config["parameters"] = sweep_params

    # create the sweep_id
    sweep_id = wandb.sweep(sweep_config, project=f"{params.data_name}_sweep")
    # run the sweep agent
    wandb.agent(sweep_id, function=main, count=10)

    # main()