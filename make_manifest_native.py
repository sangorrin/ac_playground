#!/usr/bin/env python3
"""
Build a Coqui-TTS JSONL manifest with pre-computed features:
- Pre-aligned phoneme IDs (phones_16ms/*.npy)
- Speaker d-vectors (VCTK_refs_16K_embeds/*.npy)
- Audio files (LJS_accented_16K/*.wav)
- Text (LJSpeech/metadata.csv)

Output format per line:
{
  "audio_file": "LJS_accented_16K/LJ001-0001_p334.wav",
  "text": "Printing, in the only sense...",
  "speaker_name": "p334",
  "phoneme_file": "phones_16ms/LJ001-0001_p334.npy",
  "d_vector_file": "VCTK_refs_16K_embeds/p334.npy"
}

Usage:
  python make_manifest_native.py \
    --ljs-root LJSpeech \
    --audio-dir LJS_accented_16K \
    --phones-dir phones_16ms \
    --dvec-dir VCTK_refs_16K_embeds \
    --output data/manifest_native.jsonl \
    --workers 32
"""

import argparse
import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

def load_ljs_text(meta_csv: Path) -> dict:
    """Load LJS metadata.csv (pipe-delimited, no quoting)"""
    meta = {}
    with meta_csv.open("r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if not row or len(row) < 2:
                continue
            utt_id = row[0].strip()
            # Prefer normalized text (row[2]), fallback to raw (row[1])
            text = (row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip())
            if utt_id and text:
                meta[utt_id] = text
    return meta

def process_wav(args_tuple) -> Optional[dict]:
    """Process single audio file and gather all feature paths"""
    wav_path, ljs_meta, phones_dir, dvec_dir = args_tuple

    # Extract utterance ID and speaker from filename: LJ001-0001_p334.wav
    stem = wav_path.stem  # LJ001-0001_p334
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None

    base_id, speaker = parts  # base_id=LJ001-0001, speaker=p334

    # Lookup text from LJS metadata
    text = ljs_meta.get(base_id)
    if not text:
        return None

    # Check all required feature files exist
    phone_file = phones_dir / f"{stem}.npy"
    dvec_file = dvec_dir / f"{speaker}.npy"

    if not phone_file.exists():
        return None
    if not dvec_file.exists():
        return None

    return {
        "audio_file": str(wav_path),
        "text": text,
        "speaker_name": speaker,
        "phoneme_file": str(phone_file),
        "d_vector_file": str(dvec_file),
    }

def main():
    parser = argparse.ArgumentParser(description="Generate manifest with phonemes and d-vectors")
    parser.add_argument("--ljs-root", type=Path, required=True, help="Path to LJSpeech/")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Path to LJS_accented_16K/")
    parser.add_argument("--phones-dir", type=Path, required=True, help="Path to phones_16ms/")
    parser.add_argument("--dvec-dir", type=Path, required=True, help="Path to VCTK_refs_16K_embeds/")
    parser.add_argument("--output", type=Path, default=Path("data/manifest_native.jsonl"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    # Load LJSpeech metadata
    meta_csv = args.ljs_root / "metadata.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_csv}")

    print(f"[1/4] Loading LJSpeech metadata from {meta_csv}")
    ljs_meta = load_ljs_text(meta_csv)
    print(f"      Loaded {len(ljs_meta)} transcripts")

    # Collect all audio files
    print(f"[2/4] Scanning audio files in {args.audio_dir}")
    wav_files = sorted(args.audio_dir.glob("*.wav"))
    print(f"      Found {len(wav_files)} wav files")

    # Prepare processing arguments
    process_args = [
        (wav, ljs_meta, args.phones_dir, args.dvec_dir)
        for wav in wav_files
    ]

    # Process in parallel
    print(f"[3/4] Processing with {args.workers} workers")
    results = []
    ok_count = 0
    skip_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_wav, arg): arg[0] for arg in process_args}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                ok_count += 1
            else:
                skip_count += 1

            if (ok_count + skip_count) % 1000 == 0:
                print(f"      Processed: {ok_count + skip_count}/{len(wav_files)}")

    print(f"      ✓ Valid: {ok_count}, ✗ Skipped: {skip_count}")

    # Write manifest
    print(f"[4/4] Writing manifest to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✓ Done! Manifest contains {len(results)} entries")
    print(f"  Output: {args.output}")

    # Sample validation
    if results:
        print("\nSample entry:")
        print(f"  {json.dumps(results[0], indent=2)}")

if __name__ == "__main__":
    main()
