#!/usr/bin/env python3
# Usage:
#   python /workspace/mfa_prepare.py \
#       --ljs-root LJSpeech \
#       --accented-wav-dir LJS_accented_16K \
#       --out-corpus mfa/corpus_ljs_accented \
#       --workers 32 --link hard

import argparse, csv, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

def read_meta(meta_path: Path) -> dict:
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

def _one(wav: Path, out_dir: Path, meta: dict, link_mode: str, overwrite: bool) -> Tuple[str,str]:
    try:
        utt = wav.stem                          # e.g., LJ005-0047_p334
        base_utt = utt.split("_", 1)[0]         # -> LJ005-0047 (metadata lookup)

        # NEW: derive speaker id from suffix; fall back to "spk" if none
        spk = utt.split("_")[-1] if "_" in utt else "spk"
        spk_dir = out_dir / spk                 # e.g., mfa/corpus_ljs_accented/p334
        spk_dir.mkdir(parents=True, exist_ok=True)

        # write/link inside the speaker folder
        lab = spk_dir / f"{utt}.lab"
        dst = spk_dir / f"{utt}.wav"

        text = meta.get(base_utt)
        if text is None:
            return ("MISS", utt)

        if overwrite or not dst.exists():
            ensure_link(wav, dst, link_mode)

        if overwrite or not lab.exists():
            lab.write_text(text.strip() + "\n", encoding="utf-8")

        return ("OK", utt)
    except Exception as e:
        return ("ERR", f"{wav.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ljs-root", required=True, help="Original LJSpeech root (has metadata.csv)")
    ap.add_argument("--accented-wav-dir", required=True, help="Your LJS_accented_16K dir (flat)")
    ap.add_argument("--out-corpus", required=True, help="Corpus dir for MFA (wav + .lab)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--link", choices=["hard","symlink","copy"], default="hard",
                    help="How to place WAVs into corpus (hard is fastest)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    meta = read_meta(Path(args.ljs_root) / "metadata.csv")
    in_dir = Path(args.accented_wav_dir)
    out_dir = Path(args.out_corpus); out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))  # flat dir = faster than rglob
    if not wavs:
        print("[ERR] No wavs found.")
        return

    ok = miss = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, w, out_dir, meta, args.link, args.overwrite) for w in wavs]
        for fut in as_completed(futs):
            s, _ = fut.result()
            if s == "OK": ok += 1
            elif s == "MISS": miss += 1
            else: err += 1

    print(f"[DONE] corpus={out_dir}  ok={ok}  missing_text={miss}  errors={err}  total={len(wavs)}")

if __name__ == "__main__":
    main()
