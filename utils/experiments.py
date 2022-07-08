# Utilities for experiment runs: output, config, logging

import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from shutil import copy
from types import SimpleNamespace
from typing import List, Tuple, Union

import torch

OUTPUT_DIR = os.environ["OUTPUT_DIR"]


def load_config(experiment_name: str = None) -> SimpleNamespace:
    """Loads the configuration for the specified experiment.

    Parameters
    ----------
    experiment_name : str, optional
        Experiment to load the config for. If None, the config at the project root is used

    Returns
    -------
    SimpleNamespace
        Model and training configuration
    """
    # helper hook
    def dict_to_namespace(d):
        return SimpleNamespace(**d)

    if experiment_name is None:
        # use config.json at root
        config_path = "./config.json"
    else:
        # get config from experiment dir
        config_path = os.path.join(
            os.getcwd(), OUTPUT_DIR, experiment_name, "config.json"
        )

    with open(config_path, "r") as f:
        config = json.load(f, object_hook=dict_to_namespace)

    # set data path from .env variable
    config.data_path = os.environ["DATA_PATH"]

    return config


def setup_loggers(
    command: str, log_path: os.PathLike = None, logger_names: List[str] = None
):
    """Setup one or multiple loggers such that logs are stored at the given path.

    Parameters
    ----------
    command : str
        Command name of for which the loggers are being set up
    log_path : os.PathLike, optional
        File the loggers should log to, by default None in which case the file is created at `./logs/command/yy-mm-dd_hh-mm-ss.log`
    logger_names : List[str], optional
        Names of loggers that should log to `log_path`, by default None in which case the command is used as a logger name
    """
    if log_path is None:
        # create default log_path from command and directories along it
        log_path = f"./logs/{command}/{datetime.now().strftime('%Y-%m-%d')}_{datetime.now().strftime('%H-%M-%S')}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # set up file_handler to log_path
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    if logger_names is None or len(logger_names) == 0:
        # use default logger for command
        logger_names = [command]

    # add file handler to specified loggers
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)

    logging.info(f"Added file_handler to file {log_path} to loggers {logger_names}")


def store_config(config: SimpleNamespace) -> None:
    """Store the specified config at `config.exp_dir/config.json`.

    Parameters
    ----------
    config : SimpleNamespace
        Model and training configuration
    """
    # dumb config into directory
    with open(os.path.join(config.exp_dir, "config.json"), "w") as config_file:
        json.dump(config_to_dict(config), fp=config_file, indent=4, sort_keys=True)


def create_experiment(config: SimpleNamespace):
    """Create an output directory for a new experiment.

    Parameters
    ----------
    config : SimpleNamespace
        Model and training configuration

    Returns
    -------
    SimpleNamespace
        Config with updated experiement name and experiment directory
    """
    # construct base experimment name
    experiment_name = f"{datetime.now().strftime('%m-%d')}_{config.name}"

    # get previous experiment runs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiments = os.listdir(OUTPUT_DIR)

    # get previous run numbers
    prev_runs = list(
        map(
            lambda x: int(x[(len(experiment_name) + 1) :])
            if experiment_name == x[:-4]  # _ + three digit experiment number
            else -1,
            experiments,
        )
    )

    # get number for new experiment run
    if len(prev_runs) == 0:
        num = 0
    else:
        num = max(prev_runs) + 1

    # create next experiment run
    run_name = experiment_name + f"_{num:03d}"
    config.run_name = run_name

    exp_dir = os.path.join(OUTPUT_DIR, run_name)

    # make directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "optimizer"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "val"), exist_ok=True)

    # if resuming from checkpoint, load model and optimizers
    if config.checkpoint != 0:
        old_dir = config.exp_dir

        checkpoint_model = os.path.join(
            old_dir, "models", f"model_{config.checkpoint}.pth"
        )
        checkpoint_optim = os.path.join(
            old_dir, "optimizer", f"optim_{config.checkpoint}.pth"
        )

        # copy model and optimizer to exp_dir
        copy(checkpoint_model, os.path.join(exp_dir, "models"))
        copy(checkpoint_optim, os.path.join(exp_dir, "optimizer"))

    # store new directory path
    config.exp_dir = exp_dir

    return config


def config_to_dict(config: SimpleNamespace) -> dict:
    """Convert a config `SimpleNamespace` object to dict.

    Parameters
    ----------
    config : SimpleNamespace
        Model and training configuration

    Returns
    -------
    dict
        Dict for model and training configuration
    """
    config_dict = config.__dict__

    # NOTE: if there are any nested namespaces, convert them to dicts first
    try:
        config_dict["dcd_opts"] = config.dcd_opts.__dict__
        config_dict["weight_dict"] = config.weight_dict.__dict__
        config_dict["lr_scheduler"] = config.lr_scheduler.__dict__
    except AttributeError:
        pass

    return config_dict


def save_optimizer(optimizer, epoch: int, config: SimpleNamespace, name: str = None):
    """Save the current state_dict() of the optimizer.

    Parameters
    ----------
    optimizer : _type_
        Optimizer to save state
    epoch : int
        Current epoch
    config : SimpleNamespace
        Model and training configuration
    name : str
        Used to store the optimizer as `optim_{name}.pth`, defaults to None. If None, `epoch` will be used as a name
    """
    if name is None:
        name = epoch

    # NOTE: not tested for multiple GPUs
    state_dict = deepcopy(optimizer.state_dict())

    torch.save(
        {"epoch": epoch, "optim_state_dict": state_dict},
        f"{config.exp_dir}/optimizer/optim_{name}.pth",
    )


def save_model(model, epoch: int, config: SimpleNamespace, name: str = None):
    """Save the current state_dict() of the model.

    Parameters
    ----------
    model : _type_
        Model to save state
    epoch : int
        Current epoch
    config : SimpleNamespace
        Model and training configuration
    name : str
        Used to store the model as `model_{name}.pth`, defaults to None. If None, `epoch` will be used as a name

    """
    if name is None:
        name = epoch

    if isinstance(model, torch.nn.DataParallel):
        # handle multiple GPUs
        state_dict = deepcopy(model.module.state_dict())
    else:
        state_dict = deepcopy(model.state_dict())

    torch.save(
        {"epoch": epoch, "model_state_dict": state_dict},
        f"{config.exp_dir}/models/model_{name}.pth",
    )


def get_optimizer(
    name_or_epoch: Union[str, int], config: SimpleNamespace
) -> Tuple[int, dict]:
    """Load the optimizer state for the specified name or epoch.

    Parameters
    ----------
    name_or_epoch : Union[str, int]
        Used to retrieve the optimizer file as `optim_{name_or_epoch}.pth`
    config : SimpleNamespace
        Model and training configuration

    Returns
    -------
    Tuple[int, dict]
        epoch, optim_state_dict
    """
    optim_file = f"{config.exp_dir}/optimizer/optim_{name_or_epoch}.pth"
    stored_dict = torch.load(optim_file, map_location="cpu")
    return stored_dict["epoch"], stored_dict["optim_state_dict"]


def get_model(
    name_or_epoch: Union[str, int], config: SimpleNamespace
) -> Tuple[int, dict]:
    """Load the model state for the specified name or epoch.

    Parameters
    ----------
    name_or_epoch : Union[str, int]
        Used to retrieve the model file as `model_{name_or_epoch}.pth`
    config : SimpleNamespace
        Model and training configuration

    Returns
    -------
    Tuple[int, dict]
        epoch, model_state_dict
    """
    model_file = f"{config.exp_dir}/models/model_{name_or_epoch}.pth"
    stored_dict = torch.load(model_file, map_location="cpu")
    return stored_dict["epoch"], stored_dict["model_state_dict"]

def get_num_params_total(model):
    model_parameters = model.parameters()
    model_total_params = sum(p.numel() for p in model_parameters)
    return model_total_params

def get_num_params(model):
    # return num of total parameters
    dense_parameters = model.final_conv.parameters()
    model_parameters = model.parameters()

    dense_num_params = sum(p.numel() for p in dense_parameters)
    model_total_params = sum(p.numel() for p in model_parameters)
    coarse_num_params = model_total_params - dense_num_params

    ret = {
        "coarse": coarse_num_params,
        "dense": model_total_params,
    }

    return ret
