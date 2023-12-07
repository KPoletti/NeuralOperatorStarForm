""" 
Help functions for converting between 
"""
import yaml
from pathlib import Path
import json
from neuralop import LpLoss, H1Loss


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
        "FieldwiseAggregatorLoss",
        "loss_fn",
        "losses",
        "mapping",
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


def set_Loss(params) -> LpLoss | H1Loss:
    if params.loss_name == "LpLoss":
        return LpLoss(d=params.d, p=2, reduce_dims=(0, 1))
    elif params.loss_name == "H1Loss":
        return H1Loss(d=params.d, reduce_dims=(0, 1))
    else:
        raise ValueError(f"Loss {params.loss_name} not implemented")


def loadConfig(path):
    """Load a config.yaml file.

    Args:
        path (str): path to the config.yaml file

    Returns:
        json: the config file as a json object
    """
    config = yaml.safe_load(Path(path).read_text())
    # remove desc from config sub-dictionary
    for key in config:
        if isinstance(config[key], dict):
            if "desc" in config[key]:
                del config[key]["desc"]
                config[key] = config[key]["value"]
    return config


def convertDictToNamespace(config):
    """Convert a json object to a namespace.

    Args:
        config (dict): the config file as a json object

    Returns:
        namespace: the config file as a namespace
    """

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return Namespace(**config)


def yamlToNamespace(path: str):
    """Loads yaml file directly to  namespace

    Args:
        path (str): path to yaml
    """
    config = loadConfig(path)
    config = convertDictToNamespace(config)
    config.loss_fn = set_Loss(config)
    return config


def testLoadConfig():
    """Test loadConfig() function."""
    config = loadConfig("wandb/latest-run/files/config.yaml")
    print(config)
    print(type(config))


def testConvertDictToNamespace():
    """Test convertDictToNamespace() function."""
    config = loadConfig("wandb/latest-run/files/config.yaml")
    config = convertDictToNamespace(config)
    print(config.TRAIN_PATH)
    print(type(config))


if __name__ == "__main__":
    print("##############################################")
    print(" # Testing loadConfig() function #")
    print("##############################################")
    testLoadConfig()
    print("##############################################")
    print(" # Testing convertDictToNamespace() function #")
    print("##############################################")
    testConvertDictToNamespace()
