#!/usr/bin/env python3
# Usage:
#   python /workspace/mfa_prepare.py \
#       --ljs-root LJSpeech \
#       --accented-wav-dir LJS_accented_16K \
#       --out-corpus mfa/corpus_ljs_accented \
#       --workers 32 --link hard
#
# For ARCTIC:
#   python /workspace/mfa_prepare.py \
#       --arctic-root /dataset/data/ARCTIC \
#       --wav-dir /dataset/data/ARCTIC_16k \
#       --out-corpus /dataset/data/ARCTIC_mfa_corpus \
#       --workers 32 --link hard

import argparse, csv, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

def read_meta(meta_path: Path) -> dict:
    """Read LJSpeech metadata.csv"""
    meta = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):  # ignore quotes
            if not row or len(row) < 2:
                continue
            # row = [ID, Transcription, Normalized?]
            utt = row[0].strip()
            text = (row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip())
            if utt and text:
                meta[utt] = text
    return meta

def read_arctic_transcripts(arctic_root: Path) -> dict:
    """Read ARCTIC transcript files from all speakers"""
    meta = {}
    for speaker_dir in arctic_root.iterdir():
        if not speaker_dir.is_dir() or speaker_dir.name in ["suitcase_corpus"]:
            continue

        transcript_dir = speaker_dir / "transcript"
        if not transcript_dir.exists():
            continue

        for txt_file in transcript_dir.glob("*.txt"):
            utt_id = txt_file.stem  # e.g., arctic_a0001
            try:
                text = txt_file.read_text(encoding="utf-8").strip()
                if text:
                    meta[utt_id] = text
            except Exception as e:
                print(f"Warning: Failed to read {txt_file}: {e}")

    print(f"[ARCTIC] Loaded {len(meta)} transcripts from {arctic_root}")
    return meta

def ensure_link(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # fall through to symlink if hardlink not allowed
    if mode in ("symlink", "hard"):
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    # last resort: copy bytes
    dst.write_bytes(src.read_bytes())

def _one(wav: Path, out_dir: Path, meta: dict, link_mode: str, overwrite: bool, dataset_type: str) -> Tuple[str,str]:
    try:
        if dataset_type == "ljspeech":
            # LJSpeech format: LJ005-0047_p334.wav
            utt = wav.stem                          # e.g., LJ005-0047_p334
            base_utt = utt.split("_", 1)[0]         # -> LJ005-0047 (metadata lookup)
            # Derive speaker id from suffix; fall back to "spk" if none
            spk = utt.split("_")[-1] if "_" in utt else "spk"

        elif dataset_type == "arctic":
            # ARCTIC format: /path/wavs_16k/arctic_a0001_ABA.wav (flat with speaker suffix)
            utt = wav.stem                          # e.g., arctic_a0001_ABA
            if "_" in utt:
                base_utt = "_".join(utt.split("_")[:-1])  # -> arctic_a0001 (remove speaker suffix)
                spk = utt.split("_")[-1] # Speaker from filename suffix (ABA, ASI, etc.)
            else:
                base_utt = utt
                spk = "unknown"

        else:
            return ("ERR", f"Unknown dataset type: {dataset_type}")

        spk_dir = out_dir / spk                 # e.g., mfa/corpus/ABA
        spk_dir.mkdir(parents=True, exist_ok=True)

        # write/link inside the speaker folder
        lab = spk_dir / f"{utt}.lab"
        dst = spk_dir / f"{utt}.wav"

        text = meta.get(base_utt)
        if text is None:
            return ("MISS", f"{base_utt} (from {utt})")

        if overwrite or not dst.exists():
            ensure_link(wav, dst, link_mode)

        if overwrite or not lab.exists():
            lab.write_text(text.strip() + "\n", encoding="utf-8")

        return ("OK", utt)
    except Exception as e:
        return ("ERR", f"{wav.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    # LJSpeech options
    ap.add_argument("--ljs-root", help="Original LJSpeech root (has metadata.csv)")
    ap.add_argument("--accented-wav-dir", help="Your LJS_accented_16K dir (flat)")

    # ARCTIC options
    ap.add_argument("--arctic-root", help="Original ARCTIC root (has speaker/transcript/ folders)")
    ap.add_argument("--wav-dir", help="Directory with 16kHz WAV files (for ARCTIC: speaker subfolders)")

    # Common options
    ap.add_argument("--out-corpus", required=True, help="Corpus dir for MFA (wav + .lab)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--link", choices=["hard","symlink","copy"], default="hard",
                    help="How to place WAVs into corpus (hard is fastest)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Determine dataset type and load metadata
    if args.ljs_root and args.accented_wav_dir:
        # LJSpeech mode
        meta = read_meta(Path(args.ljs_root) / "metadata.csv")
        in_dir = Path(args.accented_wav_dir)
        wavs = sorted(in_dir.glob("*.wav"))  # flat dir
        dataset_type = "ljspeech"

    elif args.arctic_root and args.wav_dir:
        # ARCTIC mode
        meta = read_arctic_transcripts(Path(args.arctic_root))
        in_dir = Path(args.wav_dir)
        wavs = sorted(in_dir.rglob("*.wav"))  # speaker subfolders
        dataset_type = "arctic"

    else:
        print("[ERR] Must specify either (--ljs-root + --accented-wav-dir) or (--arctic-root + --wav-dir)")
        return

    out_dir = Path(args.out_corpus)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wavs:
        print(f"[ERR] No wavs found in {in_dir}")
        return

    print(f"[INFO] Dataset: {dataset_type}, Files: {len(wavs)}, Metadata entries: {len(meta)}")

    ok = miss = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, w, out_dir, meta, args.link, args.overwrite, dataset_type) for w in wavs]
        for fut in as_completed(futs):
            s, msg = fut.result()
            if s == "OK":
                ok += 1
            elif s == "MISS":
                miss += 1
                if miss <= 5:  # Show first few missing files
                    print(f"[MISS] {msg}")
            else:
                err += 1
                if err <= 5:  # Show first few errors
                    print(f"[ERR] {msg}")

    print(f"[DONE] corpus={out_dir}  ok={ok}  missing_text={miss}  errors={err}  total={len(wavs)}")

if __name__ == "__main__":
    main()
