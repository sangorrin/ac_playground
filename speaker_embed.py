#!/usr/bin/env python3
# speaker_embed.py — Extract ECAPA d-vectors for single speaker or batch (VCTK)
# Usage (single):
#   python speaker_embed.py --speaker p264 --reference data/VCTK/wav48_silence_trimmed/p264/p264_001_mic1.flac
# Usage (batch over VCTK):
#   python speaker_embed.py --batch
#
# Options:
#   --speaker | --spk     Speaker ID (required if not --batch)
#   --reference | --ref   Reference audio path (required if not --batch)
#   --outdir | --outd     Output dir for .npy (default: data/spk_embed/ecapa)
#   --batch               If set, iterate over VCTK speakers and write all

import argparse
import os
import sys
import glob
import numpy as np
import torchaudio

# SpeechBrain moved "pretrained" to "inference" in v1.0; try new then old.
try:
    from speechbrain.inference import EncoderClassifier
except Exception:
    from speechbrain.pretrained import EncoderClassifier  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="ECAPA speaker embedding extractor")
    p.add_argument("--speaker", "--spk", dest="speaker", type=str, default=None,
                   help="Speaker ID (required if not --batch)")
    p.add_argument("--reference", "--ref", dest="reference", type=str, default=None,
                   help="Reference audio path (required if not --batch)")
    p.add_argument("--outdir", "--outd", dest="outdir", type=str,
                   default="data/spk_embed/ecapa", help="Output directory")
    p.add_argument("--batch", action="store_true",
                   help="Process all VCTK speakers in data/VCTK/wav48_silence_trimmed")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_wav_mono(path: str):
    """Load audio as mono tensor [1, T]."""
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def encode_file(clf: EncoderClassifier, wav_path: str) -> np.ndarray:
    wav, _ = load_wav_mono(wav_path)
    with torch_no_grad():
        emb = clf.encode_batch(wav).squeeze().detach().cpu().numpy()
    return emb


class torch_no_grad:
    def __enter__(self):
        import torch
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
    def __exit__(self, exc_type, exc, tb):
        import torch
        torch.set_grad_enabled(self.prev)


def save_embedding(emb: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    np.save(out_path, emb)


def find_vctk_ref(spk_dir: str) -> str | None:
    """Pick a canonical reference file for a VCTK speaker."""
    spk = os.path.basename(spk_dir.rstrip("/"))
    # Prefer mic1
    cand = sorted(glob.glob(os.path.join(spk_dir, f"{spk}_*_mic1.flac")))
    if not cand:
        # Fallback: any flac
        cand = sorted(glob.glob(os.path.join(spk_dir, "*.flac")))
    return cand[0] if cand else None


def run_single(clf: EncoderClassifier, speaker: str, reference: str, outdir: str) -> int:
    if not os.path.isfile(reference):
        print(f"[ERR] reference not found: {reference}", file=sys.stderr)
        return 2
    try:
        emb = encode_file(clf, reference)
    except Exception as e:
        print(f"[ERR] failed to encode {reference}: {e}", file=sys.stderr)
        return 3
    out_path = os.path.join(outdir, f"{speaker}.npy")
    save_embedding(emb, out_path)
    print(f"[OK] {speaker} {emb.shape} -> {out_path}")
    return 0


def run_batch_vctk(clf: EncoderClassifier, outdir: str) -> int:
    root = "data/VCTK/wav48_silence_trimmed"
    if not os.path.isdir(root):
        print(f"[ERR] VCTK directory not found: {root}", file=sys.stderr)
        return 2
    speakers = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not speakers:
        print(f"[ERR] no speakers found under {root}", file=sys.stderr)
        return 3
    ok, fail = 0, 0
    for spk_dir in speakers:
        spk = os.path.basename(spk_dir)
        ref = find_vctk_ref(spk_dir)
        if not ref:
            print(f"[SKIP] {spk}: no reference FLAC found", file=sys.stderr)
            fail += 1
            continue
        try:
            emb = encode_file(clf, ref)
            out_path = os.path.join(outdir, f"{spk}.npy")
            save_embedding(emb, out_path)
            print(f"[OK] {spk} {emb.shape} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {spk}: {e}", file=sys.stderr)
            fail += 1
    print(f"[DONE] saved: {ok}, failed/skipped: {fail}")
    return 0 if ok > 0 else 4


def main():
    args = parse_args()
    # Load encoder once (CPU)
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    if args.batch:
        return sys.exit(run_batch_vctk(clf, args.outdir))
    # Single mode requires both speaker and reference
    if not args.speaker or not args.reference:
        print("[ERR] --speaker/--spk and --reference/--ref are required when not using --batch",
              file=sys.stderr)
        return sys.exit(1)
    return sys.exit(run_single(clf, args.speaker, args.reference, args.outdir))


if __name__ == "__main__":
    main()
