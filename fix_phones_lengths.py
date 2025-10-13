#!/usr/bin/env python3
# Usage:
#   python scripts/fix_phones_lengths.py \
#     --phones-dir phones_20ms \
#     --f0-dir f0_20ms \
#     --out-dir phones_20ms_fix \
#     --workers 32
#
# It pads/trims each phones array to match F0 length.
# "sil" id is read from phoneme_map.json (default = 0 if missing).

import argparse, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

def load_sil_id(phones_dir: Path) -> int:
    m = phones_dir / "phoneme_map.json"
    if m.exists():
        mp = json.loads(m.read_text("utf-8"))
        # tolerate various keys ("sil", "sp", "silence"), prefer "sil"
        for k in ("sil", "silence", "sp", "spn"):
            if k in mp: return int(mp[k])
        # fallback: smallest id
        return int(min(mp.values()))
    return 0

def _fix_one(utt, phf, f0f, out_dir, sil_id):
    try:
        ph = np.load(phf, mmap_mode="r").astype(np.int16)  # [Tph]
        f0 = np.load(f0f, mmap_mode="r")                   # [Tf0]
        Tph, Tf0 = int(ph.shape[0]), int(f0.shape[0])
        if Tph == Tf0:
            out = ph
        elif Tph < Tf0:
            pad = np.full(Tf0 - Tph, sil_id, dtype=np.int16)
            out = np.concatenate([ph, pad], axis=0)
        else:
            out = ph[:Tf0]
        np.save(out_dir / f"{utt}.npy", out)
        return 1, 0
    except Exception as e:
        return 0, 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phones-dir", required=True)
    ap.add_argument("--f0-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 8)*2))
    args = ap.parse_args()

    ph_dir = Path(args.phones_dir)
    f0_dir = Path(args.f0_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # copy phoneme_map.json through
    m = ph_dir / "phoneme_map.json"
    if m.exists():
        (out_dir / "phoneme_map.json").write_text(m.read_text("utf-8"), encoding="utf-8")

    sil_id = load_sil_id(ph_dir)

    ph_files = sorted([p for p in ph_dir.glob("*.npy") if p.name != "phoneme_map.npy"])
    tasks = []
    for phf in ph_files:
        utt = phf.stem
        f0f = f0_dir / f"{utt}.npy"
        if f0f.exists():
            tasks.append((utt, phf, f0f, out_dir, sil_id))

    ok = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_fix_one, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            a, b = fut.result(); ok += a; err += b
            if i % 5000 == 0:
                print(f"[PROGRESS] fixed {i}/{len(tasks)}")
    print(f"[DONE] wrote={ok} errors={err} out={out_dir}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    main()
