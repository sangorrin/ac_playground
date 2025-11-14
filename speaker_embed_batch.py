#!/usr/bin/env python3
# python speaker_embed_batch.py \
#   --wav-dir data/VCTK_refs_16K \
#   --out-dir VCTK_refs_16K_embeds \
#   --batch-size 64 --device cuda

import argparse
from pathlib import Path
import numpy as np
import torch, torchaudio
from torch.utils.data import DataLoader, Dataset

# Use the modern speechbrain interface
from speechbrain.inference import EncoderClassifier


class WavSet(Dataset):
    def __init__(self, wav_dir, target_sr=16000):
        self.paths = sorted([p for p in Path(wav_dir).rglob("*.wav")])
        self.sr = target_sr
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        wav, sr = torchaudio.load(str(p))

        # Enforce 16 kHz strictly — do NOT resample silently.
        if sr != self.sr:
            raise ValueError(f"Sample rate mismatch for {p.name}: got {sr}, expected {self.sr}. "
                             "Resample your corpus beforehand.")

        # Enforce mono audios.
        if wav.shape[0] != 1:
            raise ValueError(f"{p.name} is multi-channel ({wav.shape[0]} ch). Provide mono 16 kHz WAVs.")

        # normalize loudness
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak  # peak normalization to ~[-1, 1]

        # Extra safety: strip NaN/Inf if any (rare corrupted files)
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        return p, wav.squeeze(0)  # [T]


def collate(batch):
    paths, waves = zip(*batch)
    lens = torch.tensor([w.shape[0] for w in waves], dtype=torch.int64)
    T = int(lens.max().item())
    pad = torch.zeros(len(waves), T, dtype=torch.float32)

    for i, w in enumerate(waves):
        pad[i, : w.shape[0]] = w

    return paths, pad, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, help="Dir tree of wavs (e.g., VCTK_refs_16K)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = WavSet(args.wav_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device}
    )

    count = 0
    with torch.inference_mode():
        for paths, wavs, lens in dl:
            wavs = wavs.to(args.device)
            emb = ecapa.encode_batch(wavs)  # [B, 1, 192]
            emb = emb.squeeze(1)  # [B, 192] - FIX: remove singleton dim
            emb = torch.nn.functional.normalize(emb, dim=-1)

            for i, p in enumerate(paths):
                npy = out / (Path(p).stem + ".npy")
                if npy.exists() and not args.overwrite:
                    continue
                # Now saves (192,) instead of (1, 192)
                np.save(npy, emb[i].detach().cpu().numpy().astype(np.float32))
                count += 1

    print(f"[DONE] Generated {count} embeddings at {out}")


if __name__ == "__main__":
    main()
