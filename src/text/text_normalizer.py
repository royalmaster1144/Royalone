#!/usr/bin/env python3
"""
text_normalizer.py (VCTK-specific pairing for .txt in txt/* and audio in wav48* or wav48_silence_trimmed)

Usage:
  python src/text_normalizer.py --vctk_root data/VCTK/ --out_csv data/meta/normalized_transcripts.csv --drop_empty
"""
import argparse, csv, os, re
from pathlib import Path
from num2words import num2words
from unidecode import unidecode
from tqdm import tqdm

PUNCT_RE = re.compile(r"[\"#$()*+,\-./:;<=>?@\[\]^_`{|}~]")
NUM_RE = re.compile(r"\d+[,\d]*\.?\d*")

ABBREVIATIONS = {
    "mr.": "mister", "mrs.": "misses", "dr.": "doctor",
    "can't": "cannot", "won't": "will not", "n't": " not",
    "&": "and", "@": "at", "%": " percent",
}

def expand_abbreviations(text):
    text = text.lower()
    for k,v in ABBREVIATIONS.items():
        text = text.replace(k, f" {v} ")
    return text

def normalize_number(match):
    token = match.group(0).replace(",", "")
    try:
        if "." in token:
            parts = token.split(".")
            left = int(parts[0]) if parts[0] != "" else 0
            right = parts[1]
            left_words = num2words(left)
            right_words = " ".join(list(right))
            return f"{left_words} point {right_words}"
        else:
            return num2words(int(token))
    except Exception:
        return token

def normalize_text(text):
    if text is None:
        return ""
    text = unidecode(text)
    text = re.sub(r"\[.*?\]|\{.*?\}|\(.*?\)|<.*?>", " ", text)
    text = text.strip().lower()
    text = expand_abbreviations(text)
    text = NUM_RE.sub(normalize_number, text)
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_pairs_vctk(vctk_root: Path):
    vctk_root = Path(vctk_root)
    pairs = []

    # Look for transcript files under txt/<speaker>/*.txt
    txt_root = vctk_root / "txt"
    txt_files = []
    if txt_root.exists():
        txt_files = list(txt_root.rglob("*.txt"))

    # If no txt folder, fallback to any .txt in tree
    if not txt_files:
        txt_files = list(vctk_root.rglob("*.txt"))

    # audio directories to look into (wav48, wav16, wav48_silence_trimmed, etc)
    audio_dirs = []
    for candidate in vctk_root.iterdir():
        if candidate.is_dir() and "wav" in candidate.name.lower():
            audio_dirs.append(candidate)

    # also include nested wav dirs
    audio_dirs += list(vctk_root.rglob("*wav*"))

    audio_extensions = [".wav", ".flac", ".flac.wav"]  # support flac too

    # build a map of basename (without ext) -> full audio path
    audio_map = {}
    for ad in audio_dirs:
        for ext in audio_extensions:
            for f in ad.rglob(f"*{ext}"):
                audio_map[f.stem] = str(f)

    # Now iterate transcript files and try to find matching audio by basename
    for tpath in tqdm(txt_files, desc="Scanning transcripts"):
        try:
            raw = tpath.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        # name like p225_001.txt -> basename p225_001
        base = tpath.stem
        # if the transcript files are named differently (e.g., prompts.txt) attempt line parsing
        if base.lower() in ("prompts", "txt", "prompts.txt", "transcripts"):
            # try line-by-line mapping: "p225_001.wav|text" or "p225_001 text"
            for line in raw.splitlines():
                line=line.strip()
                if not line: continue
                if "|" in line:
                    fname, txt = line.split("|",1)
                elif "\t" in line:
                    fname, txt = line.split("\t",1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts)==2:
                        fname, txt = parts
                    else:
                        continue
                stem = Path(fname).stem
                audio_path = audio_map.get(stem)
                if audio_path:
                    speaker = Path(audio_path).parent.name
                    pairs.append((audio_path, txt.strip(), speaker))
            continue

        # normal case: per-file transcript file
        audio_path = audio_map.get(base)
        if audio_path:
            speaker = Path(audio_path).parent.name
            pairs.append((audio_path, raw, speaker))
        else:
            # sometimes txt files are in speaker folders as p225/p225_001.txt -> we can search by prefix
            # try matching by basename prefix
            for key, path in audio_map.items():
                if key.startswith(base) or base.startswith(key):
                    speaker = Path(path).parent.name
                    pairs.append((path, raw, speaker))
                    break

    # final fallback: pair all audio files with empty text
    if not pairs:
        for stem, path in audio_map.items():
            pairs.append((path, "", Path(path).parent.name))

    return pairs

def write_csv(records, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["wav_path", "text", "speaker_id"])
        for wav_path, text, spk in records:
            writer.writerow([wav_path, text, spk])

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--vctk_root", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--drop_empty", action="store_true")
    args=p.parse_args()

    vroot = Path(args.vctk_root)
    print("Searching in:", vroot)
    pairs = find_pairs_vctk(vroot)
    print("Found pairs:", len(pairs))

    normalized=[]
    for wav, raw_text, spk in tqdm(pairs, desc="Normalizing"):
        norm = normalize_text(raw_text)
        normalized.append((wav, norm, spk))

    if args.drop_empty:
        normalized = [r for r in normalized if r[1].strip()]

    write_csv(normalized, args.out_csv)
    print("Wrote:", args.out_csv)
    print("Preview:")
    for row in normalized[:10]:
        print(row)

if __name__=="__main__":
    main()
