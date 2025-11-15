#!/usr/bin/env python3
# Usage:
#   python check_features_20ms.py \
#     --wav-dir /workspace/augmented_data/wavs_16k \
#     --phones-dir /workspace/augmented_data/mfa_alignments \
#     --f0-dir /workspace/augmented_data/f0_features \
#     --spk-embeds-dir /workspace/augmented_data/speaker_embeddings \
#     --report /workspace/sanity_report.csv \
#     --limit 0 \
#     --workers 32
#
# (recommended env knobs)
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# ulimit -n 8192

import argparse, json, os, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

def speaker_from_utt(utt: str) -> str:
    return utt.split("_")[-1] if "_" in utt else ""

def _check_one(utt, wf, phf, f0f, spk_dir, sr_expected, hop_s, max_pid):
    issues = []

    # existence checked by caller, but double-guard
    if not (wf.exists() and phf.exists() and f0f.exists()):
        issues.append(("missing_file", f"wav={wf.exists()} f0={f0f.exists()}"))
        return utt, issues

    # phones: mmap for fast header access, quick value checks
    try:
        ph = np.load(phf, mmap_mode="r")
        if ph.ndim != 1:
            issues.append(("phones_shape", str(ph.shape)))
        if ph.dtype != np.int16:
            issues.append(("phones_dtype", str(ph.dtype)))
        T_ph = int(ph.shape[0])
        # Check for NaN or Inf (should never happen in int16, but check anyway)
        if np.isnan(ph.astype(np.float32)).any():
            issues.append(("phones_has_nan", f"count={np.isnan(ph.astype(np.float32)).sum()}"))
        if max_pid is not None:
            # fast range check via memmap ops
            pmin, pmax = int(ph.min()), int(ph.max())
            if pmin < 0 or pmax > max_pid:
                issues.append(("phones_out_of_range", f"min={pmin} max={pmax} max_pid={max_pid}"))
        # Check for zero-length
        if T_ph == 0:
            issues.append(("phones_empty", "length=0"))
        # Check average phoneme ID value
        if T_ph > 0:
            ph_mean = float(np.mean(ph))
            # If mean is too close to 0, might indicate too many silence tokens
            # Typical datasets should have diverse phoneme usage
            if ph_mean < 1.0:
                issues.append(("phones_mean_low", f"mean={ph_mean:.2f}"))
    except Exception as e:
        issues.append(("phones_load", str(e)))
        T_ph = 0

    # f0: check the shape and the lack of NaN/Inf values
    try:
        f0 = np.load(f0f, mmap_mode="r", allow_pickle=True)
        if f0.ndim != 1:
            issues.append(("f0_shape", str(f0.shape)))
        T_f0 = int(f0.shape[0])
        # Check for NaN or Inf values
        if np.isnan(f0).any():
            issues.append(("f0_has_nan", f"count={np.isnan(f0).sum()}"))
        if np.isinf(f0).any():
            issues.append(("f0_has_inf", f"count={np.isinf(f0).sum()}"))
        # Check for zero-length
        if T_f0 == 0:
            issues.append(("f0_empty", "length=0"))
        # Check for unrealistic F0 values (human voice: 50-500 Hz typical, allow 0 for unvoiced)
        f0_nonzero = f0[f0 > 0]
        if len(f0_nonzero) > 0:
            if f0_nonzero.min() < 50 or f0_nonzero.max() > 1000:
                issues.append(("f0_out_of_range", f"min={f0_nonzero.min():.1f} max={f0_nonzero.max():.1f}"))
        # Check average F0 value (for voiced frames)
        if len(f0_nonzero) > 0:
            f0_mean = float(np.mean(f0_nonzero))
            # Typical human voice: male ~120Hz, female ~220Hz, children ~300Hz
            if f0_mean < 70 or f0_mean > 400:
                issues.append(("f0_mean_unusual", f"mean={f0_mean:.1f}Hz"))
    except Exception as e:
        issues.append(("f0_load", str(e)))
        T_f0 = 0

    # wav header-only info (no decode)
    try:
        info = sf.info(str(wf))
        sr = info.samplerate
        if sr_expected and sr != sr_expected:
            issues.append(("wav_sr", f"{sr} != {sr_expected}"))
        dur_s = info.frames / float(sr) if sr > 0 else 0.0
        # Check for suspiciously short/long audio
        if dur_s < 0.1:
            issues.append(("wav_too_short", f"{dur_s:.3f}s"))
        if dur_s > 30.0:
            issues.append(("wav_too_long", f"{dur_s:.1f}s"))
        # Check for mono
        if info.channels != 1:
            issues.append(("wav_not_mono", f"channels={info.channels}"))
    except Exception as e:
        issues.append(("wav_info", str(e)))
        dur_s = 0.0

    # hop estimates & length checks
    if T_f0 > 0 and dur_s > 0:
        hop_est_f0 = dur_s / T_f0
        if abs(hop_est_f0 - hop_s) > 0.003:  # ~±3 ms tolerance
            issues.append(("f0_hop_mismatch", f"~{hop_est_f0*1000:.2f} ms"))
    if T_ph > 0 and dur_s > 0:
        hop_est_ph = dur_s / T_ph
        if abs(hop_est_ph - hop_s) > 0.003:
            issues.append(("phones_hop_mismatch", f"~{hop_est_ph*1000:.2f} ms"))

    if T_f0 > 0 and T_ph > 0 and abs(T_f0 - T_ph) > 1:
        issues.append(("length_mismatch", f"f0={T_f0} phones={T_ph}"))

    # speaker embedding presence and validity
    spk = speaker_from_utt(utt)
    if spk:
        spkf = spk_dir / f"{spk}.npy"
        if not spkf.exists():
            issues.append(("missing_spk_embed", spkf.name))
        else:
            # Check speaker embedding validity
            try:
                spk_emb = np.load(spkf, mmap_mode="r")
                if spk_emb.ndim != 1:
                    issues.append(("spk_emb_shape", str(spk_emb.shape)))
                if np.isnan(spk_emb).any():
                    issues.append(("spk_emb_has_nan", f"count={np.isnan(spk_emb).sum()}"))
                if np.isinf(spk_emb).any():
                    issues.append(("spk_emb_has_inf", f"count={np.isinf(spk_emb).sum()}"))
                # Expected 192-dim ECAPA embeddings
                if len(spk_emb) != 192:
                    issues.append(("spk_emb_wrong_dim", f"expected=192 got={len(spk_emb)}"))
                # Check average value and L2 norm
                if len(spk_emb) > 0:
                    spk_mean = float(np.mean(spk_emb))
                    spk_norm = float(np.linalg.norm(spk_emb))
                    # ECAPA embeddings typically have L2 norm around 1.0 after normalization
                    if spk_norm < 0.5 or spk_norm > 2.0:
                        issues.append(("spk_emb_norm_unusual", f"norm={spk_norm:.3f}"))
                    # Mean should be close to 0 for normalized embeddings
                    if abs(spk_mean) > 0.5:
                        issues.append(("spk_emb_mean_unusual", f"mean={spk_mean:.3f}"))
            except Exception as e:
                issues.append(("spk_emb_load", str(e)))
    else:
        issues.append(("no_speaker_suffix", ""))

    return utt, issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--phones-dir", required=True)
    ap.add_argument("--f0-dir", required=True)
    ap.add_argument("--spk-embeds-dir", required=True)
    ap.add_argument("--report", default="sanity_report.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all files")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--hop_s", type=float, default=0.02, help="20 ms")
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 8) * 2),
                    help="Parallel workers (I/O bound → 2-4x vCPUs is fine)")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    ph_dir = Path(args.phones_dir)
    f0_dir = Path(args.f0_dir)
    spk_dir = Path(args.spk_embeds_dir)

    # phoneme map for range check
    pmap_path = ph_dir / "phoneme_map.json"
    max_pid = None
    if pmap_path.exists():
        pmap = json.loads(pmap_path.read_text(encoding="utf-8"))
        max_pid = max(pmap.values()) if pmap else None

    # enumerate by phones files
    ph_files = sorted([p for p in ph_dir.glob("*.npy") if p.name != "phoneme_map.npy"])
    if args.limit > 0:
        ph_files = ph_files[:args.limit]

    tasks = []
    for phf in ph_files:
        utt = phf.stem
        wf = wav_dir / f"{utt}.wav"
        f0f = f0_dir / f"{utt}.npy"
        tasks.append((utt, wf, phf, f0f, spk_dir, args.sr, args.hop_s, max_pid))

    problems = []
    # Process pool: chunk tasks to cut overhead; avoid printing inside workers.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_check_one, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            utt, issues = fut.result()
            for tag, detail in issues:
                problems.append((utt, tag, detail))
            if i % 5000 == 0:
                print(f"[PROGRESS] checked {i}/{len(tasks)}")

    # summarize ok/bad per-utt
    total = len(tasks)
    from collections import defaultdict, Counter
    per_utt_issues = defaultdict(int)
    for u, tag, _ in problems:
        per_utt_issues[u] += 1
    ok = total - len(per_utt_issues)
    bad = len(per_utt_issues)

    # write CSV
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt", "issue", "detail"])
        for row in problems:
            w.writerow(list(row))

    print(f"[SUMMARY] total={total} ok={ok} bad={bad}")
    print(f"[REPORT]  {args.report}")
    if problems:
        cnt = Counter(tag for _, tag, _ in problems)
        top = ", ".join(f"{k}:{v}" for k, v in cnt.most_common(6))
        print(f"[TOP ISSUES] {top}")
    else:
        print("[TOP ISSUES] none")

if __name__ == "__main__":
    # keep MKL/OMP from spawning inner threads
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    main()
