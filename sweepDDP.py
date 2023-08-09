"""
File for hyperparameter tuning with wandb sweep
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import src.networkUtils as myNet
import src.train as myTrain
import torch
import yaml
from src.utilsMain import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb


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
    from neuralop.training.losses import H1Loss, LpLoss

    if params.loss_name == "LpLoss":
        return LpLoss(d=params.d, p=2, reduce_dims=(0, 1))
    elif params.loss_name == "H1Loss":
        return H1Loss(d=params.d, reduce_dims=(0, 1))
    else:
        raise ValueError(f"Loss {params.loss_name} not implemented")


def main(config=None):
    import input as params

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = rank % torch.cuda.device_count()
    print(f"Device_id {device_id}")
    time.sleep(local_rank * 0.1)
    if local_rank == 0:
        run = wandb.init(project=f"{params.data_name}", config=config)
        config = wandb.config
        print(config)
    else: 
        config
    ################################################################
    # Distributed Setup
    ################################################################

    for r in range(dist.get_world_size()):
        if r == dist.get_rank():
            print(
                f"Global rank {dist.get_rank()} initialized: "
                f"local_rank = {local_rank}, "
                f"world_size = {dist.get_world_size()}",
            )
        dist.barrier()
    print(
        f"Free: {torch.cuda.mem_get_info()[0] * 10**-9:1.3f} GiB\t Avail: {torch.cuda.mem_get_info()[1] * 10**-9:1.3f} GiB"
    )
    
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

    # add the path to params
    params.path = path
    # define the loss function
    params.loss_fn = set_Loss(params)
    # Print the parameters
    paramsJSON = convertParamsToJSON(params)
    # print(json.dumps(paramsJSON, indent=4, sort_keys=True))

    # check if path exists
    time.sleep(local_rank * 0.1)
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
    model = model.to(device_id)
    DDP_model = DDP(model, device_ids=[device_id])

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
    Trainer = myTrain.Trainer(
        model=model,
        params=params,
        device=device_id,
        save_every=50,
        ckp_path=f"results/{params.path}/models/{params.NN}_snapshot.pt",
    )
    Trainer.train(trainLoader, testLoader, output_encoder, sweep=True)
    os.remove(f"results/{params.path}/models/{params.NN}_snapshot.pt")

    ################################################################
    # test neural network
    ################################################################
    logger.info("........Testing neural network........")
    # test neural network
    Trainer.evaluate(
        validLoader,
        output_encoder=output_encoder,
        input_encoder=input_encoder,
    )
    dist.destroy_process_group()
    wandb.finish()
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
    # parser = argparse.ArgumentParser()
    print(f"Found {torch.cuda.device_count()} GPUs")
    ######### input parameters #########
    # parser.add_argument("--input", type=str, default="input", help="input file")
    # parser.add_argument("--N", type=int, default=params.N, help="Number of Data points")
    import input as params

    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary
    paramsDict = convertParamsToJSON(params)
    with open(f"./sweep_config_{params.data_name}.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config=config)
