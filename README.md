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

Output: Turn-structured JSONL, one conversation per line.

```bash
python build_childes_dataset.py --out data/childes_10m.jsonl
# Optional: add UK data
python build_childes_dataset.py --out data/childes_10m.jsonl --uk-dir ./Eng-UK
