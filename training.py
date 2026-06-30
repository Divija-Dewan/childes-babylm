# File: training.py
# -----------------------------
# Main script for pretraining an LM with the next-token prediction loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
from contextlib import nullcontext
import os
import math
import wandb
import gc
import pickle

from utils import get_config, setup_experiment, setup_wandb
from models import initialize_model_and_optimizers, save_epoch_checkpoint
from data_utils import load_babylm_data

def full_train_loop(cfg, model, optimizer, scheduler, dataloader):
    # Start the loop
    start_time = time()
    epoch_size = len(dataloader)
    print(epoch_size)
    for epoch in range(cfg["n_epochs"]):
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tr_metrics = train_epoch(cfg, model, optimizer, scheduler, dataloader, epoch, epoch_size, start_time)
        print(f"Epoch {epoch}; train loss: {tr_metrics['loss']}") 
        metric_path = os.path.join(cfg["logdir"], f"epoch_{epoch}_metrics.pth")
        torch.save(tr_metrics, metric_path)

        checkpoint_dir = cfg["checkpoint_dir"]
        save_epoch_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

def unpack_batch(minibatch, device):
    input_tokens = minibatch[0].to(device)
    target_tokens = minibatch[1].to(device)
    target_mask = minibatch[2].to(device)

    return input_tokens, target_tokens, target_mask

def train_epoch(cfg, model, optimizer, scheduler, dataloader, epoch, epoch_size, start_time):
    model.train()
    total_loss = 0
    total_tokens = 0
    temp_loss = 0
    temp_tokens = 0

    device = model.device

    num_steps = len(dataloader)
    for train_step, minibatch in enumerate(tqdm(dataloader)):
        input_tokens, target_tokens, target_mask = unpack_batch(minibatch, device)
        num_tokens = torch.sum(target_mask).item()
        B = input_tokens.shape[0]

        # Perform forward pass (autocast only on CUDA; full precision elsewhere)
        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            logits = model(input_tokens)['logits']
            log_probs = F.log_softmax(logits, dim=2)
            token_log_probs = torch.gather(log_probs, 2, target_tokens.unsqueeze(2)).squeeze(2)
            
        # Backward
        loss = - torch.sum(token_log_probs * target_mask) / torch.sum(target_mask)
        loss.backward()
        if cfg["gradient_clip_norm"] != -1: 
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        temp_loss += loss.item() * num_tokens
        temp_tokens += num_tokens

        if cfg["use_wandb"] and (train_step % 10 == 0 and train_step > 0):
            # Compute the steps
            steps = epoch_size * epoch + train_step
            wandb_train_epoch(
                temp_loss / temp_tokens, steps, start_time
            )

            temp_loss = 0
            temp_tokens = 0

        # Intermediate checkpoint saving spot (guard against tiny dataloaders)
        if epoch == 0 and cfg["training_type"] == "strict_small" and train_step != 0:
            one_million_steps = len(dataloader) // 10
            if one_million_steps > 0 and train_step % one_million_steps == 0:
                curr_words = f"{train_step // one_million_steps}M"
                save_epoch_checkpoint(model, optimizer, scheduler, curr_words, cfg["checkpoint_dir"])
        if epoch == 0 and cfg["training_type"] == "strict" and train_step != 0:
            one_million_steps = len(dataloader) // 100
            if one_million_steps > 0 and train_step % one_million_steps == 0 and train_step // one_million_steps < 10:
                curr_words = f"{train_step // one_million_steps}M"
                save_epoch_checkpoint(model, optimizer, scheduler, curr_words, cfg["checkpoint_dir"])

            ten_million_steps = len(dataloader) // 10
            if ten_million_steps > 0 and train_step % ten_million_steps == 0:
                curr_words = f"{10 * (train_step // ten_million_steps)}M"
                save_epoch_checkpoint(model, optimizer, scheduler, curr_words, cfg["checkpoint_dir"])

    return {"loss" : total_loss / total_tokens}
        
def wandb_train_epoch(loss, step, start_time):
    time_elapsed = (time() - start_time) / 60
    curr_dict = {
        f"train_metrics/time_elapsed" : time_elapsed,
        f"train_metrics/batch_train_loss" : loss,
    }
    wandb.log(curr_dict, step=step)

def main():
    # Setup the experiment
    cfg = get_config()

    setup_experiment(cfg)
    if cfg["use_wandb"]:
        setup_wandb(cfg)
    print("Env init")

    # Load the data first so the LR scheduler can span the ACTUAL training length
    dataloader = load_babylm_data(cfg)
    cfg["num_training_steps"] = cfg["n_epochs"] * len(dataloader)
    cfg["num_warmup_steps"] = min(cfg["num_warmup_steps"], cfg["num_training_steps"] // 10)
    print(f"Data loaded: {len(dataloader)} steps/epoch -> "
          f"{cfg['num_training_steps']} total steps, {cfg['num_warmup_steps']} warmup")

    # Load the model and optimizers
    model, optimizer, scheduler = initialize_model_and_optimizers(cfg)
    print("Models loaded")

    # Perform training
    full_train_loop(cfg, model, optimizer, scheduler, dataloader)
    

if __name__ == "__main__":
    main()
