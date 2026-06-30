# File: models.py
# ---------------
# All functions related to loading and saving models

import os
import torch
from utils import mkdir
import gc

import transformers
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
try:
    from vllm import LLM, SamplingParams   # only needed for generation/eval
except ImportError:
    LLM = SamplingParams = None            # training does not require vllm

DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')


## INITIALIZATION ##
def initialize_model_and_optimizers(cfg):
    student = initialize_model(cfg)
    optimizer = initialize_optimizer(cfg, student)
    scheduler = initialize_scheduler(cfg, student, optimizer) 
    return student, optimizer, scheduler

def initialize_model(cfg):
    # First load the student
    size = "100m" if cfg['training_type'] == 'strict' else '10m'
    config = GPT2Config.from_pretrained(f"./configs/{size}")
    student = GPT2LMHeadModel(config).to(DEVICE)
    return student

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def initialize_optimizer(cfg, student):
    lr = cfg['learning_rate']    
    decay_parameters = get_parameter_names(student, ALL_LAYERNORM_LAYERS)    
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in student.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [
                p for n, p in student.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999) 
    )

    return optimizer

def initialize_scheduler(cfg, student, optimizer):
    num_training_steps = cfg["num_training_steps"]
    num_warmup_steps = cfg["num_warmup_steps"]
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps,
                                                             num_training_steps = num_training_steps)
    return scheduler

## SAVING AND LOADING ##
def save_epoch_checkpoint(student, optimizer, scheduler, epoch, checkpoint_dir):
    # Open a folder for the round
    folder = os.path.join(checkpoint_dir, f'epoch_{epoch}')
    mkdir(folder)

    # Save the metrics and model
    torch.save(optimizer.state_dict(), os.path.join(folder, 'latest_optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(folder, 'latest_scheduler.pt'))
    torch.save(student.state_dict(), os.path.join(folder, 'latest_student.pt'))

