"""
File for hyperparameter tuning with wandb sweep
"""

import os
from src.utilsMain import *
import torch
import numpy as np
import src.networkUtils as myNet
import src.trainBangle as myTrainBangle
import src.train as myTrain
from neuralop.utils import count_model_params
import logging
import time
import wandb
import json
import yaml


def set_Loss(params):
    from neuralop import LpLoss, H1Loss

    l2loss = LpLoss(d=1, p=2)
    if params.loss_name == "LpLoss":
        return LpLoss(d=params.d, p=2, reduce_dims=(0, 1))
    elif params.loss_name == "H1Loss":
        return H1Loss(d=params.d, reduce_dims=(0, 1))
    elif params.loss_name == "weighted":

        def weighted_loss(x, y, mask, alpha=params.alpha):
            return (1 - alpha) * l2loss(x[mask], y[mask]) + alpha * l2loss(
                x[~mask], y[~mask]
            )

        return weighted_loss
    else:
        raise ValueError(f"Loss {params.loss_name} not implemented")


def main(config=None):
    import input as params

    torch.manual_seed(42)
    with wandb.init() as run:
        config = wandb.config
        # reset values based on sweep
        # TODO: make this more general
        for attr in config.keys():
            setattr(params, attr, config[attr])
        N = int((params.split[0] + params.split[1]) * params.N)
        # create output folder
        path = (
            f"SF_{params.NN}_{params.data_name}_ep{params.epochs}"
            f"_m{params.modes}_w{params.width}_S{params.S}_Lrs{params.n_layers}"
            f"_E{params.encoder}_MLP{params.use_mlp}_N{N}"
        )
        if params.factorization is None:
            fact = "None"
        # check if path exists
        if not os.path.exists(f"results/{path}"):
            os.makedirs(f"results/{path}")
            os.makedirs(f"results/{path}/logs")
            os.makedirs(f"results/{path}/models")
            os.makedirs(f"results/{path}/plots")
            os.makedirs(f"results/{path}/data")
        # SET UP LOGGING
        logging.basicConfig(
            level=os.environ.get("LOGLEVEL", params.level),
            format="%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s",
            handlers=[
                # logging.StreamHandler(),
                logging.FileHandler(f"results/{path}/logs/output.log", mode="w"),
            ],
        )
        logging.getLogger("wandb").setLevel(logging.WARNING)
        logger.info(f"Output folder for {path}")

        # add the path to params
        params.path = path
        # define the loss function
        params.loss_fn = set_Loss(params)
        logger.info("Setting Loss function to %s with dim %s", params.loss_fn, params.d)
        ################################################################
        # load data
        ################################################################
        logger.info("........Loading data........")
        dataTime = time.time()
        # trainTime, testsTime, validTime = loadData(params, isData=False)
        (
            trainLoader,
            testLoader,
            validLoader,
            input_encoder,
            output_encoder,
        ) = prepareDataForTraining(params)
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
        # params.batch_size = torch.cuda.device_count() * params.batch_size
        model = model.to(device)
        wandb.config["Model-Num-Params"] = count_model_params(model)
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
        if "angle" in params.data_name:
            Trainer = myTrainBangle.TrainerDuo(
                model=model, params=params, device=device, save_every=20
            )
        else:
            Trainer = myTrain.Trainer(
                model=model, params=params, device=device, save_every=20
            )
        logger.debug(model)
        if params.level == "DEBUG":
            print(model)
        Trainer.train(trainLoader, testLoader, output_encoder)

        ################################################################
        # test neural network
        ################################################################
        logger.info("........Testing neural network........")
        # test neural network
        if "RNN" in params.NN or params.NN == "UNet":
            Trainer.evaluate_rnn(
                validLoader,
                output_encoder=output_encoder,
                input_encoder=input_encoder,
                savename="",
                do_animate=False,
            )
        else:
            Trainer.evaluate(
                validLoader,
                output_encoder=output_encoder,
                input_encoder=input_encoder,
                savename="",
                do_animate=False,
            )
        print(f"Output folder for {path}")
        os.popen(f"cp {run.dir}/config.yaml results/{params.path}/")


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
    print(f"Found {torch.cuda.device_count()} GPUs")

    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary

    main()
