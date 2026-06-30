# File: data_utils.py
# -------------------
# Function for dataset loading, construction and saving + collation functions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import math
import random
import os
import json
import warnings
from tqdm import tqdm
import pickle

TRAIN_PATH_100M = 'data/text_data/clean_train_100M'
TRAIN_PATH_10M = 'data/text_data/clean_train_10M'
DATASETS = ['bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard']


class CHILDESConversationDataset(Dataset):
    """Multi-turn CHILDES dataset built from our turn-structured JSONL.

    Mirrors FullBabyLMDataset's public interface exactly (`model_bos`,
    `model_eos`, and __getitem__ -> LongTensor([bos] + chunk + [eos])) so the
    existing collate_fn / training loop work unchanged.

    Serialization: each conversation -> "<speaker>: <text>" per turn joined by
    newlines, with an eos token appended to delimit conversations in the stream.
    The stream is then chopped into fixed `datapoint_length`-token chunks.
    """

    def __init__(self, cfg):
        size = "100m" if cfg['training_type'] == 'strict' else '10m'
        self.processor = AutoTokenizer.from_pretrained(f"./tokenizers/{size}")
        self.model_bos = self.processor.bos_token_id
        self.model_eos = self.processor.eos_token_id

        path = cfg["childes_jsonl_path"]
        chunk_size = cfg["datapoint_length"]

        stream = []
        n_conv = 0
        with open(path, 'r') as f:
            lines = f.readlines()
        print(f'Opened {path}; {len(lines)} conversations')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence >model_max_length notices
            for line in tqdm(lines):
                conv = json.loads(line)
                text = "\n".join(f"{t['speaker']}: {t['text']}" for t in conv["turns"])
                ids = self.processor(text, add_special_tokens=False)["input_ids"]
                stream.extend(ids)
                stream.append(self.model_eos)   # conversation delimiter
                n_conv += 1
        print(f'Tokenized {n_conv} conversations -> {len(stream)} tokens')

        self.data = [stream[i:i + chunk_size]
                     for i in range(0, len(stream), chunk_size)]
        print(f'Chunked into {len(self.data)} datapoints of <= {chunk_size} tokens')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor([self.model_bos] + self.data[idx] + [self.model_eos])

class FullBabyLMDataset(Dataset):

    def __init__(self, cfg):
        # First load the tokenizer
        size = "100m" if cfg['training_type'] == 'strict' else '10m'
        self.processor = AutoTokenizer.from_pretrained(f"./tokenizers/{size}")
        self.model_bos = self.processor.bos_token_id
        self.model_eos = self.processor.eos_token_id

        # Tokenize, split and reconstruct each dataset
        self.data = []
        dataset_folder = TRAIN_PATH_100M if cfg["training_type"] == "strict" else TRAIN_PATH_10M

        for dataset in DATASETS:
            # Load all text in dset
            dataset_path = os.path.join(dataset_folder, f'{dataset}.train')
            with open(dataset_path, 'r') as f:
                all_text = ' '.join(f.readlines())
            print(f'Opened {dataset_path}')

            # Process full text into tokens
            tokenized_dataset = self.processor(text=[all_text])['input_ids'][0]
            print(f'Tokenized {dataset_path}; {len(tokenized_dataset)} tokens total')

            # Chunk and add
            chunk_size = cfg["datapoint_length"]
            num_chunks = math.ceil(len(tokenized_dataset) / chunk_size)
            for curr_chunk in tqdm(range(num_chunks)):
                start = curr_chunk * chunk_size
                end = (curr_chunk+1) * chunk_size
                chunk_tokens = tokenized_dataset[start:end]
                self.data.append(chunk_tokens)
            print(f"Chunked {dataset_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor([self.model_bos] + self.data[idx] + [self.model_eos])

## General utilities ##
def load_babylm_data(cfg):
    # Get the overall BabyLM dataset to extract data from (behavior may vary)
    use_childes = cfg.get("data_source") == "childes_jsonl"
    cache_dir = 'data/text_data/cached_train'
    os.makedirs(cache_dir, exist_ok=True)
    if use_childes:
        num_words = "100M" if cfg["training_type"] == "strict" else "10M"
        filename = os.path.join(cache_dir, f'train_childes_gpt2_{num_words}.pkl')
    else:
        num_words = "100M" if cfg["training_type"] == "strict" else "10M"
        filename = os.path.join(cache_dir, f'train_gpt2_{num_words}.pkl')

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            full_babylm_dset = pickle.load(f)
    else:
        full_babylm_dset = CHILDESConversationDataset(cfg) if use_childes \
            else FullBabyLMDataset(cfg)
        with open(filename, 'wb') as f:
            pickle.dump(full_babylm_dset, f)

    collate_fn = get_collate_fn(full_babylm_dset.model_eos)
    dataloader = DataLoader(full_babylm_dset, batch_size=cfg["batch_size"],
                            shuffle=True, collate_fn=collate_fn)
    return dataloader

def get_collate_fn(model_eos):
    def collate_fn(batch):
        tokens = pad_sequence([item for item in batch], padding_value=model_eos, batch_first=True)
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        target_mask = input_tokens != model_eos
        target_mask[:, 0] = 1

        return input_tokens, target_tokens, target_mask
    return collate_fn
    
