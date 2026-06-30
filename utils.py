# File: utils.py
# --------------
# Minor utility functions

import argparse
import wandb
import os
import yaml
import random
import numpy as np
import torch

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapoint_length', type=int,
                        help="The length of a datapoint, regardless of the underlying dataset")
    parser.add_argument('--training_type', type=str, choices=["strict", "strict_small"])

    # Training hyperparameters
    parser.add_argument('--n_epochs', type=int,
                        help="Max number of epochs to train for a given round")
    parser.add_argument('--batch_size', type=int,
                        help="Batch size for training")

    parser.add_argument('--learning_rate', type=float,
                        help="The learning rate for training")
    parser.add_argument('--weight_decay', type=float,
                        help="The weight decay for training")
    parser.add_argument('--num_training_steps', type=int,
                        help="Maximum number of training steps for the scheduler")
    parser.add_argument('--num_warmup_steps', type=int,
                        help="Number of warmup steps for the scheduler")
    parser.add_argument('--gradient_clip_norm', type=float,
                        help="Gradient clipping value, if used")
    
    # Experiment hyperparameters
    parser.add_argument('--seed', type=int,
                        help="Random seed for reproducibility")
    parser.add_argument('--base_folder', type=str,
                        help="The name of the folder holding all experimentation data")
    parser.add_argument('--experiment_name', type=str,
                        help="The name of the current experiment")
    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to log experimental results")
    parser.add_argument('--wandb_project_name', type=str,
                        help="The project name for wandb")
    parser.add_argument('--wandb_experiment_name', type=str,
                        help="The experiment name for wandb")

    args = parser.parse_args()
    config = construct_config(args)
    return config

def setup_experiment(cfg):
    # Set the seed for reproducibility
    if cfg["seed"] == -1:
        cfg["seed"] = random.randint(0, 1000000)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

    # Make the relevant folders for the current experiment
    cfg["expdir"] = os.path.join(
        cfg["base_folder"],
        cfg["experiment_name"]
    )
    cfg["checkpoint_dir"] = os.path.join(cfg["expdir"], 'checkpoints')
    cfg["logdir"] = os.path.join(cfg["expdir"], 'logging')
    mkdir(cfg["expdir"])
    mkdir(cfg["checkpoint_dir"])
    mkdir(cfg["logdir"])

    with open(os.path.join(cfg["logdir"], "exp_cfg.yaml"), 'w') as cfg_file:
        yaml.dump(cfg, cfg_file)

def setup_wandb(cfg):
    wandb_input = {"entity" : "mog29",
                   "name" : cfg["wandb_experiment_name"], 
                   "project" : cfg["wandb_project_name"]}
    wandb.init(**wandb_input)

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def construct_config(args):
    base_path = os.path.join('config.yaml')
    cfg = load_yaml(base_path)

    # Iterate over arguments and replace new arguments with defaults in the config
    args_dict = args.__dict__
    for key, value in args_dict.items():
        if value is None:
            continue
        cfg[key] = value

    return cfg
