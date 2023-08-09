import argparse
import json
import logging
import os
import time

import numpy as np
import src.networkUtils as myNet
import src.train as myTrain
import torch
import torch.distributed as dist
from src.utilsMain import prepareDataForTraining
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb


def convertParamsToDict(params_obj):
    """
    Converts the params to a dictionary for wandb
    """
    params_dict = {}
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
    for key, value in vars(params_obj).items():
        # skip the levels that are not needed
        if key in unneeded:
            continue
        params_dict[key] = value
    return params_dict


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


def main(params):
    ################################################################
    # Distributed Setup
    ################################################################
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    print(f"Start running basic DDP Neural Operator on rank {rank}.")
    # initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = rank % torch.cuda.device_count()
    print(f"Device_id {device_id}")

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
    ################################################################
    # Wandb Setup
    ################################################################

    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary
    paramsJSON = convertParamsToJSON(params)
    run = wandb.init(project=params.data_name, config=paramsJSON)

    ################################################################
    # PATH
    ################################################################
    # grab the ending of the density file with .pt
    N = int((params.split[0] + params.split[1]) * params.N)
    # create output folder
    path = (
        f"SF_{params.NN}_{params.data_name}_ep{params.epochs}"
        f"_m{params.modes}_w{params.width}_S{params.S}_Lrs{params.n_layers}"
        f"_E{params.encoder}_MLP{params.use_mlp}_N{N}"
    )
    params.path = path
    # check if path exists
    time.sleep(local_rank * 0.1)
    if not os.path.exists(f"results/{path}"):
        os.makedirs(f"results/{path}")
        os.makedirs(f"results/{path}/logs")
        os.makedirs(f"results/{path}/models")
        os.makedirs(f"results/{path}/plots")
        os.makedirs(f"results/{path}/data")
    params.lr = params.lr * dist.get_world_size()
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

    ################################################################
    # create neural network
    ################################################################
    logging.info("........Creating neural network........")
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
    logging.info(f"Neural network created in {time.time() - netTime} seconds")

    ################################################################
    # train neural network
    ################################################################
    logging.info("........Training neural network........")
    # train neural network
    Trainer = myTrain.Trainer(
        model=DDP_model,
        params=params,
        device=device_id,
        save_every=50,
        ckp_path=f"results/{params.path}/models/{params.NN}_snapshot.pt",
    )
    Trainer.train(trainLoader, testLoader, output_encoder)
    os.remove(f"results/{params.path}/models/{params.NN}_snapshot.pt")
    ################################################################
    # test neural network
    ################################################################
    logging.info("........Testing neural network........")
    # test neural network
    Trainer.evaluate(
        validLoader,
        output_encoder=output_encoder,
        input_encoder=input_encoder,
        savename="ValidationData",
    )
    dist.destroy_process_group()
    wandb.finish()
    print(f"Output folder for {path}")
    # run.finish()


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    ################################################################
    # Parser
    ################################################################
    # change the directory to main.py's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    print(f"Found {torch.cuda.device_count()} GPUs")
    ######### input parameters #########
    parser.add_argument("--input", type=str, default="input", help="input file")
    params = __import__(parser.parse_args().input)
    parser.add_argument("--N", type=int, default=params.N, help="Number of Data points")
    main(params)
