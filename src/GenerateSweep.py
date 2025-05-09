import json
import os.path
import sys

import yaml

import wandb

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import input_FNO2d_Turb as params  # noqa: E402


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


def main():
    with open("private/wandb_api_key.txt") as f:
        wandb_api_key = f.readlines()[0]
    wandb.login(key=wandb_api_key)
    # convert params to dictionary
    paramsDict = convertParamsToJSON(params)
    # define the sweep_config
    sweep_config = {
        "method": "random",
        "metric": {"name": "Test-Loss-L2", "goal": "minimize"},
    }
    # create the sweep parameters
    sweep_params = {
        "modes": {"distribution": "int_uniform", "min": 4, "max": 48},
        "width": {
            "distribution": "int_uniform",
            "min": 12,
            "max": 128,
        },
        "lr": {"min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [5, 10, 15, 25]},
        "loss_name": {"values": ["LpLoss", "H1Loss"]},
        "n_layers": {"min": 2, "max": 6},
        "skip_type": {"values": ["linear", "soft-gating", "identity"]},
    }
    # create a list of the sweep parameters
    list(sweep_params.keys())
    # loop over paramsDict and to sweep params if they are not in sweep_params
    for key, value in paramsDict.items():
        if key not in sweep_params.keys():
            sweep_params[key] = {"value": value}
    # add sweep parameters to sweep_config
    sweep_config["parameters"] = sweep_params
    # Save the sweep config to a yaml file
    with open(f"sweep_config_{params.data_name}_{params.NN}.yaml", "w") as f:
        yaml.dump(sweep_config, f)


if __name__ == "__main__":
    main()
