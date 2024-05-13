import argparse
import logging
import os
import time

import numpy as np
import src.networkUtils as myNet
import src.train as myTrain
import src.trainBangle as myTrainBangle
from src.utilsMain import prepareDataForTraining
from src.parameterUtils import convertParamsToJSON
from neuralop.utils import count_model_params  # pyright: ignore
import torch
import torch.nn as nn
import wandb
from src.dissipative_utils import (
    sample_uniform_spherical_shell,
    linear_scale_dissipative_target,
)


def main(params):
    ################################################################
    # Wandb Setup
    ################################################################
    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary
    paramsJSON = convertParamsToJSON(params)
    if "MNO" in params.NN:
        params.out_dim = 1
        params.dissloss = nn.MSELoss(reduction="mean")
        params.sampling_fn = sample_uniform_spherical_shell
        params.target_fn = linear_scale_dissipative_target
        params.radius = 156.25 * params.S  # radius of inner ball
        params.scale_down = 0.1  # rate at which to linearly scale down inputs
        params.loss_weight = (
            0.01 * params.S**2
        )  # normalized by L2 norm in function space
        params.radii = [
            params.radius,
            (525 * params.S) + params.radius,
        ]  # inner and outer radii, in L2 norm of function space
    run = wandb.init(project=params.data_name, config=paramsJSON)

    ################################################################
    # PATH
    ################################################################
    # grab the ending of the density file with .pt
    N = int((params.split[0] + params.split[1]) * params.N)
    # create output folder
    fact = params.factorization
    if params.factorization is None:
        fact = "None"
    path = (
        f"SF_{params.NN}_{params.data_name}_ep{params.epochs}"
        f"_m{params.modes}_w{params.width}_S{params.S}_Lrs{params.n_layers}"
        f"_E{params.encoder}_MLP{params.use_mlp}_N{N}_Fact{fact}"
    )
    params.path = path
    if params.diss_reg:
        params.dissloss = params.loss_fn
        params.sampling_fn = sample_uniform_spherical_shell
        params.target_fn = linear_scale_dissipative_target
        params.radii = (
            params.radius,
            (525 * params.S) + params.radius,
        )  # inner and outer radii, in L2 norm of function space
    # check if path exists
    if not os.path.exists(f"results/{path}"):
        os.makedirs(f"results/{path}")
        os.makedirs(f"results/{path}/logs")
        os.makedirs(f"results/{path}/models")
        os.makedirs(f"results/{path}/plots")
        os.makedirs(f"results/{path}/data")

    ################################################################
    # SET UP LOGGING
    ################################################################
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", params.level),
        format="%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s",
        handlers=[
            # logging.StreamHandler(),
            logging.FileHandler(f"results/{path}/logs/output.log", mode="w"),
        ],
    )
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.info(f"Output folder for {path}")

    ################################################################
    # load data
    ################################################################
    logging.info("........Loading data........")
    dataTime = time.time()
    # trainTime, testsTime, validTime = loadData(params, isData=False)
    (
        trainLoader,
        testLoader,
        validLoader,
        input_encoder,
        output_encoder,
    ) = prepareDataForTraining(params, params.S)
    logging.info(f"Data loaded in {time.time() - dataTime} seconds")
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ################################################################
    # create neural network
    ################################################################
    logging.info("........Creating neural network........")
    netTime = time.time()
    # create neural network
    model = myNet.initializeNetwork(params)
    wandb.config["Model-Num-Params"] = count_model_params(model)

    # params.batch_size = torch.cuda.device_count() * params.batch_size
    model = model.to(device)
    if params.encoder:
        output_encoder.cuda()
        input_encoder.cuda()
    # log the model to debug
    logging.info(f"Neural network created in {time.time() - netTime} seconds")

    ################################################################
    # train neural network
    ################################################################
    logging.info("........Training neural network........")
    # train neural network
    if "angle" in params.data_name:
        Trainer = myTrainBangle.TrainerDuo(
            model=model, params=params, device=device, save_every=20
        )
    else:
        Trainer = myTrain.Trainer(
            model=model, params=params, device=device, save_every=20
        )
    Trainer.train(trainLoader, testLoader, output_encoder)

    ################################################################
    # test neural network
    ################################################################
    logging.info("........Testing neural network........")
    # test neural network
    if "RNN" in params.NN or "UNet" in params.NN:
        Trainer.evaluate_RNN(
            validLoader,
            output_encoder=output_encoder,
            input_encoder=input_encoder,
            savename="ValidationData",
        )
    else:
        Trainer.evaluate(
            validLoader,
            output_encoder=output_encoder,
            input_encoder=input_encoder,
            savename="ValidationData",
        )
    print(f"Output folder for {path}")
    os.popen(f"cp {run.dir}/config.yaml results/{params.path}/")  # pyright: ignore
    run.finish()  # pyright: ignore


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    ################################################################
    # Parser
    ################################################################
    # change the directory to main.py's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    print(f"Found {torch.cuda.device_count()} GPUs")
    ######### input parameters #########
    parser.add_argument(
        "--input", type=str, default="input", help="input file")
    params = __import__(parser.parse_args().input)
    parser.add_argument("--N", type=int, default=params.N, help="Number of Data points")
    main(params)
