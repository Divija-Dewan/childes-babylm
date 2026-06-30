#!/usr/bin/env python3
"""
build_childes_dataset.py
========================
Build a BabyLM-compliant (strict-small, <=10M words) multi-turn dialogue
dataset from English CHILDES, output as turn-structured JSONL.

Why this design
---------------
* CHILDES is distributed in CHAT (.cha) format, NOT plain text. We use
  `pylangacq` to parse it and strip CHAT annotations (retracings [/], errors
  [* ...], xxx/yyy unintelligible, &=gestures, %mor/%gra tiers, etc.).
* BabyLM strict-small allows custom corpora but caps the TRAINING SET at 10M
  words (and total exposure at 100M = 10M x 10 epochs). So we shuffle
  transcripts with a fixed seed and accumulate until the word budget is hit.
* Both speakers are kept (child + caregiver); one transcript == one
  conversation, preserving multi-turn structure.
* Speaker ROLE is read from each file's @Participants header (Target_Child,
  Mother, ...), because the child is NOT always coded "CHI" (e.g. "MAR").

Data sources / access reality
-----------------------------
* Eng-NA: open master bundle 0-Eng-NA-MOR.zip is auto-downloaded (~91 MB,
  7,800+ transcripts). This alone is ~15-20M words -- more than the 10M cap.
* Eng-UK: NOT openly downloadable via a stable URL (TalkBank gates the UK
  collections behind an auth/agreement modal). To include it, download the
  UK corpora manually from https://childes.talkbank.org/access/Eng-UK/ ,
  unzip them into a folder, and pass that folder via --uk-dir.

Usage
-----
    pip install pylangacq
    python build_childes_dataset.py --out childes_10m.jsonl
    # to add manually-downloaded UK data:
    python build_childes_dataset.py --out childes_10m.jsonl --uk-dir ./Eng-UK

Output: one JSON object per line:
    {"source":"CHILDES","region":"Eng-NA","collection":"Brown",
     "file":"Brown/adam01.cha","child_age_months":27.0,
     "turns":[{"speaker":"MOT","role":"caregiver","text":"what are you doing ?"},
              {"speaker":"CHI","role":"child","text":"i want the ball ."}, ...]}
"""

import argparse
import json
import random
import re
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

warnings.simplefilter("ignore")  # rustling emits rare mor/word alignment notes

try:
    import pylangacq
except ImportError:
    sys.exit("Missing dependency. Run:  pip install pylangacq")

ENG_NA_MASTER = "https://childes.talkbank.org/access/Eng-NA/0-Eng-NA-MOR.zip"

# Role classification from the @Participants header `role` field.
CAREGIVER_ROLE_HINTS = (
    "Mother", "Father", "Parent", "Grandmother", "Grandfather", "Grandparent",
    "Aunt", "Uncle", "Caretaker", "Caregiver", "Babysitter", "Nurse", "Adult",
    "Teacher", "Relative", "Sibling", "Brother", "Sister",
)
JUNK_TOKENS = {"xxx", "yyy", "www", "0", "CLITIC"}
KEEP_PUNCT = {".", "!", "?", ",", ";", ":"}
WORDISH = re.compile(r"[A-Za-z0-9]")


def classify_role(role: str) -> str:
    if not role:
        return "other"
    if "Target_Child" in role or role == "Child":
        return "child"
    if any(h in role for h in CAREGIVER_ROLE_HINTS):
        return "caregiver"
    return "other"


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  cached  {dest.name}")
        return True
    try:
        print(f"  fetch   {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "babylm-childes/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP    {url}  ({e})")
        if dest.exists():
            dest.unlink()
        return False


def fetch_eng_na(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    cha_root = cache / "cha"
    out_dir = cha_root / "Eng-NA"
    if out_dir.exists() and any(out_dir.rglob("*.cha")):
        print("  cached  Eng-NA extracted")
        return cha_root
    na_zip = cache / "Eng-NA-MOR.zip"
    if download(ENG_NA_MASTER, na_zip):
        with zipfile.ZipFile(na_zip) as z:
            z.extractall(cha_root)  # zip already contains an Eng-NA/ prefix
    return cha_root


def read_chat_no_mor(path: Path):
    """Parse a .cha after stripping %mor/%gra/etc. dependent tiers.

    The MOR-tagged CHILDES bundles make rustling cross-check the %mor tier
    against the word tier; on the (frequent) misalignments it drops the
    utterance's tokens entirely. Removing %-tiers recovers that text while
    keeping the main *speaker tiers and @headers we actually use.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    stripped = "\n".join(ln for ln in raw.split("\n") if not ln.startswith("%"))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".cha", delete=False,
                                      encoding="utf-8")
    try:
        tmp.write(stripped)
        tmp.close()
        return pylangacq.read_chat(tmp.name, strict=False)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def clean_utterance(utt) -> str:
    words = []
    for tok in (utt.tokens or ()):
        w = (tok.word or "").strip()
        if not w or w in JUNK_TOKENS:
            continue
        if WORDISH.search(w) or w in KEEP_PUNCT:   # drop +"/. and other CHAT codes
            words.append(w)
    text = " ".join(words)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)    # tidy space before punct
    return text.strip()


def count_words(text: str) -> int:
    return sum(1 for t in text.split() if WORDISH.search(t))


def parse_file(path: Path, region: str, merge_consecutive: bool):
    """Parse one .cha -> (conversation dict, word_count) or (None, 0)."""
    reader = read_chat_no_mor(path)

    # code -> role map from header participants
    role_map = {}
    try:
        for p in reader.headers()[0].participants:
            role_map[p.code] = classify_role(p.role)
    except Exception:  # noqa: BLE001
        pass

    age = None
    try:
        ages = reader.ages()
        if ages and ages[0] is not None:
            age = round(ages[0].in_months(), 1)  # in_months is a method
    except Exception:  # noqa: BLE001
        pass

    turns = []
    for utt in reader.utterances():
        code = (utt.participant or "").upper()
        text = clean_utterance(utt)
        if not text:
            continue
        role = role_map.get(code, "other")
        if merge_consecutive and turns and turns[-1]["speaker"] == code:
            turns[-1]["text"] += " " + text
        else:
            turns.append({"speaker": code, "role": role, "text": text})

    if not turns:
        return None, 0
    words = sum(count_words(t["text"]) for t in turns)
    parts = path.parts
    collection = parts[parts.index(region) + 1] if region in parts else ""
    rel = "/".join(parts[parts.index(region) + 1:]) if region in parts else path.name
    conv = {
        "source": "CHILDES", "region": region, "collection": collection,
        "file": rel, "child_age_months": age, "turns": turns,
    }
    return conv, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="childes_10m.jsonl")
    ap.add_argument("--cache", default="childes_cache")
    ap.add_argument("--uk-dir", default=None,
                    help="folder of manually-downloaded Eng-UK .cha files")
    ap.add_argument("--word-budget", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-merge", action="store_true",
                    help="keep every utterance as its own turn")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)

    # collect (path, region) pairs
    items = []
    if not args.skip_download:
        print("[1/3] Eng-NA ...")
        cha_root = fetch_eng_na(cache)
    else:
        cha_root = cache / "cha"
    for p in sorted((cha_root / "Eng-NA").rglob("*.cha")) if (cha_root / "Eng-NA").exists() else []:
        items.append((p, "Eng-NA"))
    if args.uk_dir:
        uk = Path(args.uk_dir)
        uk_files = sorted(uk.rglob("*.cha"))
        print(f"      Eng-UK: {len(uk_files)} local transcripts from {uk}")
        for p in uk_files:
            # ensure 'Eng-UK' appears in parts for collection bookkeeping
            items.append((p, "Eng-UK" if "Eng-UK" in p.parts else uk.name))

    if not items:
        sys.exit("No .cha files found. Did the download fail? Try --uk-dir or check network.")

    print(f"[2/3] {len(items)} transcripts; shuffling (seed={args.seed})")
    random.Random(args.seed).shuffle(items)

    print(f"[3/3] writing up to {args.word_budget:,} words -> {args.out}")
    total_words = total_convs = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path, region in items:
            try:
                conv, words = parse_file(path, region, not args.no_merge)
            except Exception as e:  # noqa: BLE001
                print(f"  warn: {path.name}: {e}")
                continue
            if not conv or words == 0 or total_words + words > args.word_budget:
                continue
            out.write(json.dumps(conv, ensure_ascii=False) + "\n")
            total_words += words
            total_convs += 1
            if total_words >= args.word_budget * 0.999:
                break

    print(f"\nDONE  {total_convs:,} conversations  {total_words:,} words")
    print(f"      exposure: {total_words:,} x 10 epochs = {total_words*10:,} (cap 100M)")


if __name__ == "__main__":
    main()
