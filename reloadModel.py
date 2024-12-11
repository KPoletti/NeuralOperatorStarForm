import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from src.utilsMain import prepareDataForTraining
from src.networkUtils import initializeNetwork
import src.train as myTrain
from src.parameterUtils import yamlToNamespace, loadConfig
from sweep import set_Loss


def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    return model


def checkConfig(config, params):
    config_keys = config.__dict__
    config_keys.pop("loss_fn")
    for key, value in vars(params).items():
        if key in config_keys:
            assert (
                value == config.__dict__[key]
            ), f"{key} does not match \n params: {value} \n config: {config.__dict__[key]}"
    print(f"Config matches input file")


def correctParamsForNewData(params, datapath):
    params.TRAIN_PATH = f"{datapath}.pt"
    params.TIME_PATH = f"{datapath}.h5"
    params.N = pd.read_hdf(params.TIME_PATH, "table").shape[0]
    params.split = [0, 0, 0.1]
    return params


def prepareValidData(params, S):
    import src.utilsMain as um
    from src.meta_dataset import TensorDataset

    # load data
    full_data = um.loadData(params, is_data=True)
    time_data = um.loadData(params, is_data=False)
    # divide data into time step and timestep to predict
    full_data_a, full_data_u, time_data_a, time_data_u = um.timestepSplit(
        full_data, time_data, params
    )

    # permute data to correct shape
    full_data_a = um.permute(full_data_a, params, full_data_a.shape[0], S)
    full_data_u = um.permute(full_data_u, params, full_data_u.shape[0], S)

    # split data into training, testing, and validation subsets
    train_data_a, tests_data_a, valid_data_a = um.trainTestValidSplit(
        params, full_data_a
    )
    train_data_u, _, valid_data_u = um.trainTestValidSplit(params, full_data_u)

    _, _, valid_time_a = um.trainTestValidSplit(params, time_data_a)
    _, _, valid_time_u = um.trainTestValidSplit(params, time_data_u)

    print(f"Valid Data shape = {valid_data_a.shape}")
    output_encoder = None
    input_encoder = None
    if params.encoder:
        # initialize the encoder
        input_encoder, output_encoder = um.initializeEncoder(
            valid_data_a, valid_data_u, params
        )
        # normalize input and ouput data
        train_data_a = input_encoder.encode(train_data_a)
        tests_data_a = input_encoder.encode(tests_data_a)
        valid_data_a = input_encoder.encode(valid_data_a)

        train_data_u = output_encoder.encode(train_data_u)
    valid_dataset = TensorDataset(
        valid_data_a,
        valid_data_u,
        um.dfTolist(valid_time_a),
        um.dfTolist(valid_time_u),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, drop_last=True
    )
    return valid_loader, input_encoder, output_encoder


def main(path, datapath, do_animate):
    params = yamlToNamespace(f"results/{path}/config.yaml")
    modelpath = f"results/{path}/models/{params.NN}.pt"
    params.path = f"{path}/Alt_Data"
    params.use_ddp = False
    params = correctParamsForNewData(params, datapath)

    dataname = datapath.split("/")[-1]
    if not os.path.exists(f"results/{params.path}/{dataname}"):
        os.makedirs(f"results/{params.path}/{dataname}")
        os.makedirs(f"results/{params.path}/{dataname}/data")
        os.makedirs(f"results/{params.path}/{dataname}/plots")

    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    wandb.init(
        project=f"{params.data_name}_altData",
        config=loadConfig(f"results/{path}/config.yaml"),
    )
    VP = datapath.split("_VP")[-1].split("_")[0]
    axis = datapath.split("_VP")[0].split("_")[-1]
    try:
        wandb.log({"Virial": float(VP), "axis": axis})
    except ValueError:
        print("WARNING: Virial parameter could not be converted to float")
        wandb.log({"Virial": VP, "axis": axis})

    params.split = [0, 0, 0.1]
    (
        validLoader,
        input_encoder,
        output_encoder,
    ) = prepareValidData(params, params.S)
    ################################################################
    # Model
    ################################################################
    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ################################################################
    # create neural network
    ################################################################
    # create neural network
    model = initializeNetwork(params)
    # print(model.load_state_dict())
    # print(model.state_dict())
    model = loadModel(model, modelpath)
    params.path = f"{params.path}/{dataname}"

    # params.batch_size = torch.cuda.device_count() * params.batch_size
    model = model.to(device)
    if params.encoder:
        output_encoder.cuda()
        input_encoder.cuda()
    ################################################################
    # test neural network
    ################################################################
    # test neural network
    # print(model.state_dict())
    Trainer = myTrain.Trainer(model=model, params=params, device=device, save_every=20)
    if "RNN" in params.NN:
        Trainer.evaluate_rnn(
            validLoader,
            output_encoder=output_encoder,
            input_encoder=input_encoder,
            savename="ValidationData_Alt_Data",
        )
    else:
        Trainer.evaluate(
            validLoader,
            output_encoder=output_encoder,
            input_encoder=input_encoder,
            savename="ValidationData_Alt_Data",
            do_animate=do_animate,
        )


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
    # parser.add_argument("--input", type=str, default="input", help="input file")
    # params = __import__(parser.parse_args().input)
    # parser.add_argument("--N", type=int, default=params.N, help="Number of Data points")
    parser.add_argument(
        "--datapath",
        type=str,
        default="../dataToSend/TrainingData/TurbProj_SmallerVP/Turb_x_VP2.18_dN10_dt8.29_multiData",
        help="path to data",
    )
    parser.add_argument(
        "--animate",
        help="Create Animations of the whole data.",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--runpath",
        type=str,
        default="SF_FNO2d_Turb12ALL_dN10_ep2_m64_w32_S64_Lrs7_ETrue_MLPTrue_N1529_DDP",
        help="path to a previous run generated by main.py",
    )
    print(f"ani = {parser.parse_args().animate}")
    main(
        parser.parse_args().runpath,
        parser.parse_args().datapath,
        do_animate=parser.parse_args().animate,
    )
