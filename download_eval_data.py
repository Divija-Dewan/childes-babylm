#!/usr/bin/env python3
"""
download_eval_data.py
---------------------
Download the BabyLM evaluation data from OSF (project ryjfm) WITHOUT a login,
into evaluation-pipeline-2025/evaluation_data/. Handles the password-protected
EWoK zip and its nested-path quirk automatically.

Verified working: this is the exact OSF API + waterbutler-zip method tested
locally (fast_eval = ~1 MB; full_eval is larger).

Usage:
    python download_eval_data.py --which fast      # fast_eval (for checkpoints)
    python download_eval_data.py --which full      # full_eval (final model)
    python download_eval_data.py --which both
    # optional: --dest <eval repo>/evaluation_data
"""
import argparse, io, json, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

OSF_NODE = "ryjfm"
EWOK_PASSWORD = "BabyLM2025"
API = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "babylm-eval/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _folder_id(parent_url, name):
    for item in _get(parent_url)["data"]:
        if item["attributes"]["name"] == name:
            return item["attributes"]["path"].strip("/"), \
                   item["relationships"]["files"]["links"]["related"]["href"]
    raise SystemExit(f"'{name}' not found in OSF listing")


def _download_folder_zip(folder_id, out_dir: Path):
    url = f"https://files.osf.io/v1/resources/{OSF_NODE}/providers/osfstorage/{folder_id}/?zip="
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "babylm-eval/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out_dir)
    print(f"  extracted -> {out_dir}")


def _fix_ewok(split_dir: Path):
    """Extract the password-protected ewok zip and flatten its nested path."""
    for zname in ("ewok_fast.zip", "ewok_filtered.zip", "ewok.zip"):
        zpath = split_dir / zname
        if not zpath.exists():
            continue
        print(f"  extracting {zname} (password)")
        # python's zipfile supports password; use it to avoid shelling out
        with zipfile.ZipFile(zpath) as z:
            z.extractall(split_dir, pwd=EWOK_PASSWORD.encode())
        # flatten nested evaluation_data/<split>/ewok_* -> <split>/ewok_*
        nested = split_dir / "evaluation_data"
        if nested.exists():
            for ewok_dir in nested.rglob("ewok_*"):
                if ewok_dir.is_dir():
                    target = split_dir / ewok_dir.name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(ewok_dir), str(target))
            shutil.rmtree(nested, ignore_errors=True)
        print(f"  ewok ready under {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["fast", "full", "both"], default="fast")
    ap.add_argument("--dest", default="evaluation-pipeline-2025/evaluation_data",
                    type=Path)
    args = ap.parse_args()

    # navigate ryjfm -> evaluation_data
    _, eval_link = _folder_id(API, "evaluation_data")

    wanted = {"fast": ["fast_eval"], "full": ["full_eval"],
              "both": ["fast_eval", "full_eval"]}[args.which]

    for split in wanted:
        print(f"[{split}]")
        fid, _ = _folder_id(eval_link, split)
        split_dir = args.dest / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        _download_folder_zip(fid, split_dir)
        _fix_ewok(split_dir)

    # report what we got
    print("\n=== task directories present ===")
    for split in wanted:
        d = args.dest / split
        subs = sorted(p.name for p in d.iterdir() if p.is_dir())
        print(f"  {split}: {subs}")
    print("\nDone. Point the eval scripts at:", args.dest)


if __name__ == "__main__":
    main()
