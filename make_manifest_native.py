#!/usr/bin/env python3
"""
Build a Coqui-TTS JSONL manifest from:
  - LJS normalized text (metadata.csv)
  - Your accented 16k WAVs (flat folder, e.g., LJS_accented_16K)
  - ECAPA d-vectors (one .npy per speaker, e.g., VCTK_refs_16K_embeds/p334.npy)

Each output line:
{
  "audio_file": "LJS_accented_16K/LJ001-0001_p334.wav",
  "text": "Printing, in the only sense ...",
  "speaker_name": "p334",
  "d_vector_file": "VCTK_refs_16K_embeds/p334.npy"
}

Notes:
- Text is taken from LJS `metadata.csv` using the base ID before the first underscore.
  e.g., LJ001-0001_p334 -> lookup text for LJ001-0001
- Minimal validation: only checks that WAV and d-vector exist; rows missing text or d-vector are skipped.
- Parallelized with ProcessPoolExecutor for large corpora.
"""

import argparse, csv, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

def load_ljs_text(meta_csv: Path) -> dict:
    """Load LJS metadata.csv (pipe-delimited). Prefer normalized text if present."""
    m = {}
    with meta_csv.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in r:
            if not row:
                continue
            utt = row[0].strip()
            # row = [ID, Transcription, Normalized?]
            text = (row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip())
            if utt and text:
                m[utt] = text
    return m

def speaker_from_utt(utt: str) -> str:
    """Extract pXXX from LJ001-0001_p334 -> p334. Return '' if missing."""
    return utt.split("_")[-1] if "_" in utt else ""

def list_wavs_fast(wav_dir: Path):
    """Fast iterator over .wav files in a (flat) directory."""
    for de in os.scandir(wav_dir):
        if de.is_file() and de.name.endswith(".wav"):
            yield wav_dir / de.name

def _make_row(
    wav_path: Path,
    text_map: dict,
    spk_dir: Path,
) -> Optional[Tuple[dict, str]]:
    """
    Return (row, status) or None if skipped.
    status is one of: 'OK', 'MISS_TEXT', 'MISS_SPK'
    """
    try:
        utt = wav_path.stem                 # LJ001-0001_p334
        base = utt.split("_", 1)[0]         # LJ001-0001
        text = text_map.get(base, "")
        if not text:
            return None
        spk_id = speaker_from_utt(utt)
        if not spk_id:
            return None
        dvec = spk_dir / f"{spk_id}.npy"
        if not dvec.exists():
            return None
        row = {
            "audio_file": str(wav_path),
            "text": text,
            "speaker_name": spk_id,
            "d_vector_file": str(dvec),
        }
        return (row, "OK")
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ljs-root", required=True, help="Path containing LJSpeech/metadata.csv")
    ap.add_argument("--wav-dir", required=True, help="Folder with accented 16k WAVs (flat)")
    ap.add_argument("--spk-embeds-dir", required=True, help="Folder with ECAPA d-vectors pXXX.npy")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 8) * 2),
                    help="I/O-bound: 2–4x vCPUs is fine.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all wavs")
    args = ap.parse_args()

    # Keep MKL/OMP quiet for I/O-bound work
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    meta_csv = Path(args.ljs_root) / "metadata.csv"
    text_map = load_ljs_text(meta_csv)

    wav_dir = Path(args.wav_dir)
    spk_dir = Path(args.spk_embeds_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wavs = []
    for i, p in enumerate(list_wavs_fast(wav_dir), 1):
        wavs.append(p)
        if args.limit and i >= args.limit:
            break

    if not wavs:
        print(f"[ERR] No wavs found in {wav_dir}")
        return

    ok = 0
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_make_row, w, text_map, spk_dir) for w in wavs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                row, status = r
                if status == "OK":
                    rows.append(row)
                    ok += 1
            if i % 5000 == 0:
                print(f"[PROGRESS] scanned {i}/{len(wavs)}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote: {out_path}")
    print(f"[STATS] total_wavs={len(wavs)} kept={ok} dropped={len(wavs)-ok}")

if __name__ == "__main__":
    main()
