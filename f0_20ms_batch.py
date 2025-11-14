#!/usr/bin/env python3
# python f0_20ms_batch.py \
#   --in-wav-dir LJS_accented_16K \
#   --out-dir f0_20ms \
#   --workers 8
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from amfm_decompy import basic_tools as bt
from amfm_decompy import pYAAPT

def f0_yaapt_20ms(wav_path: Path):
    sig = bt.SignalObj(str(wav_path))
    assert sig.fs == 16000, f"Expected 16 kHz, got {sig.fs}"

    # YAAPT params (ms)
    pitch = pYAAPT.yaapt(
        sig,
        f0_min=50.0, f0_max=600.0,
        frame_length=40.0,   # 40 ms window (matches your STFT win_length=640 at 16 kHz)
        frame_space=20.0,    # Use frame_space for 20 ms hop.
        tda_frame_length=25.0,
        fft_length=1024
    )
    f0 = pitch.samp_values.astype(np.float32)  # 20ms hop
    # Keep unvoiced segments as 0 (do NOT convert to NaN - causes model corruption)
    # Coqui TTS expects 0 for unvoiced frames, will handle normalization internally

    # sanity: estimate hop from duration
    est_hop_ms = (len(sig.data) / sig.fs) / len(f0) * 1000
    if not (18.5 <= est_hop_ms <= 21.5):
        raise RuntimeError(f"YAAPT hop looks wrong: ~{est_hop_ms:.2f} ms (wanted ~20 ms)")

    return f0

def _one(args):
    wav, out_dir, overwrite = args
    try:
        out = out_dir / (wav.stem + ".npy")
        if out.exists() and not overwrite: return 1
        f0 = f0_yaapt_20ms(wav)
        np.save(out, f0)
        return 1
    except Exception as e:
        print(f"[ERR] {wav.name}: {e}")
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-wav-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(in_dir.rglob("*.wav"))
    tasks = [(w, out_dir, args.overwrite) for w in wavs]
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_one, tasks):
            ok += r
    print(f"[DONE] f0 files={ok}  out={out_dir}")

if __name__ == "__main__":
    main()
