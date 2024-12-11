import argparse
import logging
import os
import time

import numpy as np
import src.networkUtils as myNet
import src.train as myTrain
import src.trainBangle as myTrainBangle
import wandb
from neuralop.utils import count_model_params  # pyright: ignore
from src.dissipative_utils import (
    linear_scale_dissipative_target,
    sample_uniform_spherical_shell,
)
from src.parameterUtils import convertParamsToJSON
from src.utilsMain import prepareDataForTraining

import torch


def main(params):
    """
    Main function to train the neural network
    """
    ################################################################
    # Wandb Setup
    ################################################################
    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary
    paramsJson = convertParamsToJSON(params)
    # initialize wandb and save the config
    run = wandb.init(project=params.data_name, config=paramsJson)

    ################################################################
    # PATH
    ################################################################
    # Calculate the Number of points in training and testing
    N = int((params.split[0] + params.split[1]) * params.N)
    # create output folder
    fact = params.factorization
    if params.factorization is None:
        fact = "None"
    params.path = (
        f"SF_{params.NN}_{params.data_name}_ep{params.epochs}"
        f"_m{params.modes}_w{params.width}_S{params.S}_Lrs{params.n_layers}"
        f"_E{params.encoder}_MLP{params.use_mlp}_N{N}_Fact{fact}"
    )
    # in
    if params.diss_reg:
        params.radius = 156.25 * params.S  # radius of inner ball
        params.scale_down = 0.1  # rate at which to linearly scale down inputs
        params.loss_weight = (
            0.01 * params.S**2
        )  # normalized by L2 norm in function space
        params.dissloss = params.loss_fn
        params.sampling_fn = sample_uniform_spherical_shell
        params.target_fn = linear_scale_dissipative_target
        params.radii = (
            params.radius,
            (525 * params.S) + params.radius,
        )  # inner and outer radii, in L2 norm of function space
    # check if path exists for the results
    if not os.path.exists(f"results/{params.path}"):
        os.makedirs(f"results/{params.path}")
        os.makedirs(f"results/{params.path}/logs")
        os.makedirs(f"results/{params.path}/models")
        os.makedirs(f"results/{params.path}/plots")
        os.makedirs(f"results/{params.path}/data")
    ################################################################
    # SET UP LOGGING
    ################################################################
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", params.level),
        format="%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"results/{params.path}/logs/output.log", mode="w"),
        ],
    )
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.info("Output folder for %s", params.path)
    ################################################################
    # load data
    ################################################################
    logging.info("........Loading data........")
    data_time = time.time()
    # trainTime, testsTime, validTime = loadData(params, isData=False)
    (
        train_loader,
        test_loader,
        valid_loader,
        input_encoder,
        output_encoder,
    ) = prepareDataForTraining(params)

    logging.info(f"Data loaded in {time.time() - data_time} seconds")
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ################################################################
    # create neural network
    ################################################################
    logging.info("........Creating neural network........")
    net_time = time.time()
    # create neural network
    model = myNet.initializeNetwork(params)
    logging.debug(model)
    wandb.config["Model-Num-Params"] = count_model_params(model)

    # params.batch_size = torch.cuda.device_count() * params.batch_size
    model = model.to(device)
    if params.encoder:
        output_encoder.cuda()
        input_encoder.cuda()
    # log the model to debug
    logging.info(f"Neural network created in {time.time() - net_time} seconds")

    ################################################################
    # train neural network
    ################################################################
    logging.info("........Training neural network........")
    # train neural network
    if "angle" in params.data_name:
        trainer = myTrainBangle.TrainerDuo(
            model=model, params=params, device=device, save_every=20
        )
    else:
        trainer = myTrain.Trainer(
            model=model, params=params, device=device, save_every=20
        )
    trainer.train(train_loader, test_loader, output_encoder)

    ################################################################
    # test neural network
    ################################################################
    logging.info("........Testing neural network........")
    # test neural network
    if "RNN" in params.NN or "UNet" in params.NN:
        trainer.evaluate_rnn(
            valid_loader,
            output_encoder=output_encoder,
            input_encoder=input_encoder,
            savename="ValidationData",
        )
    else:
        trainer.evaluate(
            valid_loader,
            output_encoder=output_encoder,
            input_encoder=input_encoder,
            savename="ValidationData",
        )
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
    parser.add_argument("--input", type=str, default="input_RNN_MHD", help="input file")
    parameters = __import__(parser.parse_args().input)
    parser.add_argument(
        "--N", type=int, default=parameters.N, help="Number of Data points"
    )
    main(parameters)
