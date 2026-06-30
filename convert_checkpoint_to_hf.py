#!/usr/bin/env python3
"""
convert_checkpoint_to_hf.py
---------------------------
Convert a trained babylm-gpt2-baseline checkpoint (a raw `latest_student.pt`
state_dict + the configs/10m GPT2Config + the tokenizers/10m tokenizer) into a
self-contained HuggingFace model directory that the BabyLM evaluation pipeline
loads via `AutoModelForCausalLM.from_pretrained(...)`.

Because our model is a *vanilla* GPT2LMHeadModel, NO custom modeling files are
needed (unlike the gpt-bert baseline / hf_conversion_tutorial). This is just:
    GPT2Config -> GPT2LMHeadModel -> load_state_dict -> save_pretrained
plus copying the tokenizer alongside.

Usage
-----
    python convert_checkpoint_to_hf.py \
        --checkpoint experiments/kaggle_run/checkpoints/epoch_0/latest_student.pt \
        --config_dir babylm-gpt2-baseline/configs/10m \
        --tokenizer_dir babylm-gpt2-baseline/tokenizers/10m \
        --out hf_models/childes_gpt2_final

To convert every checkpoint (final + intermediate 1M..10M) at once, pass
--checkpoints_root <checkpoints dir> --out_root <dir>; each epoch_* folder
becomes its own HF dir (named so the eval pipeline's `chck_*M` scheme works).
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


def convert_one(checkpoint: Path, config_dir: Path, tokenizer_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)

    config = GPT2Config.from_pretrained(config_dir)
    model = GPT2LMHeadModel(config)

    state = torch.load(checkpoint, map_location="cpu")
    # Tolerate checkpoints saved as {"model": sd} or with a "module."/"_orig_mod." prefix
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and all(not k.startswith("transformer") for k in state):
        state = state["model"]
    cleaned = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # report so silent random-init never goes unnoticed
    real_missing = [k for k in missing if "attn.bias" not in k and "attn.masked_bias" not in k]
    if real_missing:
        print(f"  [WARN] {len(real_missing)} missing keys (will be random!): {real_missing[:5]}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys ignored: {unexpected[:5]}")
    if not real_missing and not unexpected:
        print("  [OK] all weights loaded from checkpoint (no random init)")

    model.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(out)
    print(f"  [SAVED] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, help="single latest_student.pt")
    ap.add_argument("--checkpoints_root", type=Path,
                    help="dir of epoch_* folders, each with latest_student.pt")
    ap.add_argument("--config_dir", type=Path, required=True)
    ap.add_argument("--tokenizer_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="output HF dir (single mode)")
    ap.add_argument("--out_root", type=Path, help="output root (batch mode)")
    args = ap.parse_args()

    if args.checkpoint:
        if not args.out:
            sys.exit("--out is required with --checkpoint")
        convert_one(args.checkpoint, args.config_dir, args.tokenizer_dir, args.out)
    elif args.checkpoints_root:
        if not args.out_root:
            sys.exit("--out_root is required with --checkpoints_root")
        for ckpt_dir in sorted(args.checkpoints_root.glob("epoch_*")):
            f = ckpt_dir / "latest_student.pt"
            if not f.exists():
                continue
            # epoch_3M -> chck_3M (eval pipeline revision scheme); epoch_0 -> main
            tag = ckpt_dir.name.replace("epoch_", "")
            name = "main" if tag == "0" else f"chck_{tag}"
            print(f"[{ckpt_dir.name}] -> {name}")
            convert_one(f, args.config_dir, args.tokenizer_dir, args.out_root / name)
    else:
        sys.exit("Provide --checkpoint or --checkpoints_root")


if __name__ == "__main__":
    main()
