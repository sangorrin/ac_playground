#!/usr/bin/env python3
# python freevc_batch.py --num 1000 --num-random-speakers=1 --workers 1 --use_cuda [--ljs_dir LJSpeech00 --vctk_dir data/VCTK_refs_16K --out_dir out_tmp]
# Nota: este script carga el modelo una sola vez y evita relanzarlo; eso ya acelera respecto al CLI. 
# Para más velocidad, mantén workers=1 o 2; más hilos no suelen ayudar en un solo GPU con este modelo.
import argparse, os, glob, random, time, sys
from pathlib import Path

def main():
    import torch
    from TTS.api import TTS
    import soundfile as sf

    ap = argparse.ArgumentParser(description="FreeVC batch: make K random speaker pairs per LJS wav.")
    ap.add_argument("--ljs_dir",   type=str, default="LJSpeech", help="Dir of LJS wavs (e.g., data/LJSpeech_16K/wavs)")
    ap.add_argument("--vctk_dir",  type=str, default="data/VCTK_refs_16K", help="Dir of VCTK ref wavs (one per speaker)")
    ap.add_argument("--out_dir",   type=str, default="out_tmp", help="Output dir for VC wavs (24 kHz)")
    ap.add_argument("--num",       type=int, default=0, help="How many LJS items to use (0 = all found)")
    ap.add_argument("--num-random-speakers", dest="k", type=int, default=1, help="Speakers per LJS item")
    ap.add_argument("--workers",   type=int, default=1, help="Parallel workers (GPU => forces 1)")
    ap.add_argument("--use_cuda",  action="store_true", help="Use CUDA if available")
    ap.add_argument("--seed",      type=int, default=42, help="Random seed for speaker sampling")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"[info] device={device}")

    t0 = time.time()
    tts = TTS("voice_conversion_models/multilingual/vctk/freevc24").to(device)
    print(f"[info] model loaded in {time.time()-t0:.2f}s")

    ljs_files = sorted(glob.glob(os.path.join(args.ljs_dir, "*.wav")))
    if args.num and args.num > 0:
        ljs_files = ljs_files[:args.num]
    if not ljs_files:
        print("[err] no LJS wavs found in:", args.ljs_dir, file=sys.stderr)
        sys.exit(1)

    spk_files = sorted(glob.glob(os.path.join(args.vctk_dir, "*.wav")))
    if not spk_files:
        print("[err] no VCTK ref wavs found in:", args.vctk_dir, file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    if device == "cuda" and args.workers != 1:
        print("[warn] use_cuda + workers>1 is unsafe; forcing workers=1")
        args.workers = 1

    total_pairs = len(ljs_files) * max(1, args.k)
    print(f"[info] LJS={len(ljs_files)} speakers={len(spk_files)} k={args.k} -> pairs={total_pairs}")

    # Try to get the model's output sample rate; default to 24000 (FreeVC24)
    out_sr = 24000
    try:
        # coqui-tts exposes output sample rate like this (defensive fallback)
        out_sr = getattr(tts, "output_sample_rate", out_sr)
    except Exception:
        pass

    # Convert pairs
    start = time.time()
    done = 0
    log_every = max(1, total_pairs // 10)

    def pick_k(files, k):
        if k <= 0:
            return []
        if k <= len(files):
            return rng.sample(files, k)  # without replacement
        # with replacement if k > #speakers
        return [rng.choice(files) for _ in range(k)]

    for i, src in enumerate(ljs_files, 1):
        uid = Path(src).stem
        refs = pick_k(spk_files, args.k)
        for ref in refs:
            spk = Path(ref).stem
            out_path = os.path.join(args.out_dir, f"{uid}_{spk}.wav")
            try:
                # Prefer API that writes to file if available
                if hasattr(tts, "voice_conversion_to_file"):
                    tts.voice_conversion_to_file(source_wav=src, target_wav=ref, file_path=out_path)
                else:
                    wav = tts.voice_conversion(source_wav=src, target_wav=ref)
                    sf.write(out_path, wav, out_sr)
            except Exception as e:
                print(f"[err] {uid} x {spk}: {e}", file=sys.stderr)
                continue
            done += 1
            if done % log_every == 0 or done == total_pairs:
                print(f"[{done}/{total_pairs}] {out_path}")

    elapsed = time.time() - start
    per = elapsed / max(1, done)
    print(f"[info] elapsed {elapsed:.2f}s for {done} files → {per:.2f}s/file")
    print(f"[info] outputs in: {args.out_dir}")

if __name__ == "__main__":
    main()
