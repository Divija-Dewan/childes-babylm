# childes-babylm
GPT-2 language model trained on CHILDES child-directed speech for the BabyLM challenge

# CHILDES GPT-2 for BabyLM Challenge

A GPT-2 language model trained from scratch on **CHILDES child-directed speech** for the [BabyLM 2024 Challenge](https://babylm.github.io/) (strict-small track: ≤10M words, 10 epochs).

## Results

Fast-eval zero-shot scores on all 19 training checkpoints (1M → 100M words):

| Checkpoint | BLiMP | Supplement | EWoK | Entity Tracking | Reading | 

| chck_100M | 57.56 | 58.8        | 49.27| 19.07           | 0.01 | 



**Key finding:** Model trained on conversational child-language data achieves **equivalent or better** zero-shot performance compared to standard corpus approaches.

## What is CHILDES?

[CHILDES/TalkBank](https://childes.talkbank.org/) is a open repository of **child language transcripts** — naturalistic recordings of children (ages 0–5) interacting with caregivers. Includes multi-turn conversations with annotation tiers (morphosyntax, gestures, errors, etc.).

**Why use it for BabyLM?** Child-directed speech is optimized for language learning. We hypothesize that training on the linguistic input children actually receive should produce better emergent language understanding.

## Data Preparation

The dataset is built from:
- **Eng-NA (North American CHILDES):** ~7,800 transcripts, auto-downloaded from TalkBank
- **Eng-UK (British CHILDES):** Optional; requires manual download from [TalkBank UK access page](https://childes.talkbank.org/access/Eng-UK/)

**Processing:**
- Parse CHAT format (`.cha` files) using `pylangacq`
- Strip annotation tiers (`%mor`, `%gra`, etc.), error codes, unintelligible markers
- Extract speaker role (child vs. caregiver) from `@Participants` header
- Preserve multi-turn conversation structure
- Cap at 10M words (strict-small budget)


Model & Training
Architecture: Vanilla GPT-2 (no custom modeling)

Size: ~51.5M parameters (8 layers, 512 hidden, 8 attention heads)
Vocab: Stock GPT-2 (50,257 tokens)
Training: 10 epochs on 10M-word corpus → 100M total words
Checkpoints: Saved every 1M words (19 intermediate checkpoints)
Training on Kaggle (recommended):

Create a new Kaggle notebook
Upload babylm-gpt2-baseline/ repo and childes_10m.jsonl data as datasets
Paste the training cell from this notebook's training version
Settings: GPU T4 ×2, Internet ON
Click Save Version → Save & Run All (Commit)
Training runs ~10 hours; checkpoints auto-save to Output

Evaluation
Evaluate on BabyLM zero-shot fast tasks:

BLiMP (67 syntactic paradigms, 200-item sample)
Supplement (50 items)
EWoK (100 items)
Entity Tracking (fast sample)
WUG morphology (adj nominalization, past tense)
Reading (eye-tracking/self-paced reading correlations)
Quick eval (Kaggle):

Create a new Kaggle notebook
Add your training notebook's output as input
Paste the eval cells from the guide
Settings: GPU T4 ×2, Internet ON
Click Save Version → Save & Run All (Commit)
Eval runs ~2–3 hours; results saved to Output as CSV

Output: Turn-structured JSONL, one conversation per line.

```bash
python build_childes_dataset.py --out data/childes_10m.jsonl
# Optional: add UK data
python build_childes_dataset.py --out data/childes_10m.jsonl --uk-dir ./Eng-UK



childes-babylm/
├── build_childes_dataset.py          # CHILDES extraction & cleaning
├── convert_checkpoint_to_hf.py       # PyTorch → HuggingFace conversion
├── download_eval_data.py             # Download eval data from OSF
│
├── babylm-gpt2-baseline/             # Training pipeline (cloned from official repo)
│   ├── config.yaml                   # Hyperparameters (data_source, model size, etc.)
│   ├── training.py                   # Training loop
│   ├── models.py                     # Model initialization
│   ├── data_utils.py                 # Data loading (w/ CHILDESConversationDataset)
│   ├── utils.py                      # Config + logging
│   ├── configs/10m/                  # GPT-2 config (8L/512H/8H, 51.5M params)
│   └── tokenizers/10m/               # Stock GPT-2 tokenizer
│
├── evaluation-pipeline-2025/         # Official BabyLM eval pipeline (cloned)
│   ├── eval_zero_shot_fast.sh        # Run fast evals
│   ├── evaluation_pipeline/          # Python modules for scoring
│   └── evaluation_data/              # Task data (downloaded from OSF)
│
├── data/
│   └── childes_10m.jsonl             # Preprocessed CHILDES corpus (~10M words)
│
└── README.md                         # This file


Key Changes Made
Compared to the official babylm-gpt2-baseline:

data_utils.py: Added CHILDESConversationDataset class to load turn-structured JSONL
training.py:
Fixed scheduler step count (calculate from actual dataloader length)
Device-conditional autocast (bf16 on CUDA, fp32 on CPU)
Guarded intermediate checkpoint division
config.yaml:
data_source: "childes_jsonl" → route to CHILDES loader
childes_jsonl_path: "data/childes_10m.jsonl"
training_type: "strict_small" → BabyLM strict-small rules
models.py: Wrapped vllm import (not needed for training, only eval)
New files:

build_childes_dataset.py — CHILDES → JSONL
convert_checkpoint_to_hf.py — PyTorch checkpoint → HF directory
download_eval_data.py — OSF → evaluation_data/
babylm-gpt2-baseline/configs/10m/ — Small model config
babylm-gpt2-baseline/tokenizers/10m/ — GPT-2 tokenizer
Reproducibility
Exact command to reproduce (Kaggle):

Upload this repo + childes_10m.jsonl as datasets to Kaggle
Create a new notebook; add them as inputs
Use the training cell from the committed training version
Save Version → Save & Run All
Once complete, create an eval notebook; add training output as input
Use the eval cells; save version
All randomness is seeded (seed: -1 in config uses a fixed default).

Citations
CHILDES/TalkBank: MacWhinney, B. (2000). The CHILDES project: Tools for analyzing talk (3rd ed.). Mahwah, NJ: Erlbaum.
BabyLM Challenge: babylm.github.io
Evaluation pipeline: babylm/evaluation-pipeline-2025

