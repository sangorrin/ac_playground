#!/usr/bin/env python3
"""
Parallel resampling to 16 kHz mono with ffmpeg.

Usage:
  python scripts/resample_to_16k_parallel.py IN_DIR OUT_DIR [--jobs N] [--delete-src]

Notes:
  - Assumes a flat IN_DIR with .wav files (no subfolders).
  - Skips files if the output already exists.
  - Writes to a temp file, then atomically renames on success.
  - If --delete-src is set, removes the original only after a successful conversion.
"""
import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
from typing import Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional


def convert_one(in_file: Path, out_dir: Path, delete_src: bool) -> Tuple[str, str]:
    """
    Returns (status, message) where status in {"OK","SKIP","ERR"}.
    """
    try:
        out_file = out_dir / in_file.name
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            return ("SKIP", f"{in_file.name}")

        # temp file in the same folder for atomic replace
        with tempfile.NamedTemporaryFile(dir=out_file.parent, suffix=".tmp.wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(in_file),
            "-ac", "1", "-ar", "16000",
            str(tmp_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            try:
                tmp_path.unlink(missing_ok=True)
            finally:
                return ("ERR", f"{in_file.name}: {res.stderr.decode(errors='ignore').strip()[:200]}")

        os.replace(str(tmp_path), str(out_file))

        if delete_src:
            try:
                in_file.unlink()
            except Exception as e:
                return ("OK", f"{in_file.name} (converted; failed to delete src: {e})")

        return ("OK", f"{in_file.name}")
    except Exception as e:
        return ("ERR", f"{in_file.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", type=Path, help="Directory with input .wav files (flat, no subdirs)")
    ap.add_argument("out_dir", type=Path, help="Output directory for 16 kHz mono wavs")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--delete-src", action="store_true", help="Delete source files after successful conversion")
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        print("No .wav files found.", file=sys.stderr)
        sys.exit(1)

    tasks = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for w in wavs:
            tasks.append(ex.submit(convert_one, w, out_dir, args.delete_src))

        iterator = tqdm(as_completed(tasks), total=len(tasks), ncols=100) if tqdm else as_completed(tasks)

        ok = skip = err = 0
        for fut in iterator:
            status, msg = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                err += 1
            if not tqdm:
                print(f"[{status}] {msg}")

    print(f"[DONE] out={out_dir}  ok={ok}  skip={skip}  err={err}")


if __name__ == "__main__":
    main()
