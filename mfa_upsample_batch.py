#!/usr/bin/env python3
# python mfa_upsample_batch.py \
#   --corpus-dir mfa/corpus_ljs_accented \
#   --dict english_us_mfa \
#   --acoustic english_mfa \
#   --out-align mfa/alignments_ljs_accented \
#   --out-frames phones_20ms \            # <-- now writes int16 phone IDs here
#   --jobs 32
import argparse, subprocess, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from textgrid import TextGrid

def list_textgrids(root: Path):
    return sorted(root.rglob("*.TextGrid"))

def extract_phone_symbols(tg_path: Path):
    try:
        tg = TextGrid.fromFile(tg_path)
        tier = next((t for t in tg.tiers if t.name and t.name.lower().startswith("phone")), None)
        if tier is None:
            return set()
        syms = set()
        for itv in tier:
            lab = (itv.mark or "").strip()
            syms.add(lab if lab else "sil")
        return syms
    except Exception:
        return set()

def build_phoneme_map(textgrids, existing: Path|None):
    # Reuse if provided
    if existing and existing.exists():
        with existing.open("r", encoding="utf-8") as f:
            pm = json.load(f)
        if pm.get("sil", None) != 0:
            raise RuntimeError("phoneme_map.json must map 'sil' to 0.")
        return pm

    uniq = set()
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        for s in ex.map(extract_phone_symbols, textgrids):
            uniq |= s
    uniq.discard("sil")
    ordered = ["sil"] + sorted(x for x in uniq if x)
    return {p:i for i,p in enumerate(ordered)}

def upsample_textgrid_to_ids(tg_path: Path, out_dir: Path, pmap: dict[str,int], sr=16000, hop=320):
    tg = TextGrid.fromFile(tg_path)
    tier = next((t for t in tg.tiers if t.name and t.name.lower().startswith("phone")), None)
    if tier is None:
        raise RuntimeError(f"No phones tier in {tg_path}")
    dur = float(getattr(tier, "maxTime", tg.maxTime))
    T = int(round(dur * sr / hop))
    if T <= 0:
        raise RuntimeError("zero frames")
    out = np.zeros(T, dtype=np.int16)  # 0 = sil
    for itv in tier:
        s = int(round(float(itv.minTime) * sr / hop))
        e = int(round(float(itv.maxTime) * sr / hop))
        s = max(0, s)
        e = min(T, max(s+1, e))  # ensure >=1 frame
        lab = (itv.mark or "").strip() or "sil"
        out[s:e] = pmap.get(lab, 0)
    np.save(out_dir / (tg_path.stem + ".npy"), out)

def _one_textgrid(args):
    tg_path, out_dir, pmap = args
    try:
        out = out_dir / (tg_path.stem + ".npy")
        if out.exists():
            return ("SKIP", tg_path.stem)
        upsample_textgrid_to_ids(tg_path, out_dir, pmap)
        return ("OK", tg_path.stem)
    except Exception as e:
        return ("ERR", f"{tg_path.name}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--dict", default="english_us_mfa", help="MFA dict name or path")
    ap.add_argument("--acoustic", default="english_mfa", help="MFA acoustic model name or path")
    ap.add_argument("--out-align", required=True)
    ap.add_argument("--out-frames", required=True, help="Output dir for 20ms phone IDs (int16)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-align-if-exists", action="store_true")
    ap.add_argument("--phoneme-map", default=None, help="Optional path to an existing phoneme_map.json to reuse")
    args = ap.parse_args()

    align_dir = Path(args.out_align); align_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.out_frames); frames_dir.mkdir(parents=True, exist_ok=True)

    # 1) MFA alignment (only if TextGrids don't exist or not skipping)
    if not args.skip_align_if_exists or not any(align_dir.rglob("*.TextGrid")):
        cmd = [
            "mfa","align", args.corpus_dir, args.dict, args.acoustic, str(align_dir),
            "-j", str(args.jobs), "--clean", "--overwrite"
        ]
        print("[MFA]", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("[MFA] Skipped (TextGrids exist)")

    # 2) Build or load phoneme_map.json (sil → 0)
    tgs = list_textgrids(align_dir)
    if not tgs:
        raise SystemExit(f"No TextGrids found under {align_dir}")
    map_path = Path(args.phoneme_map) if args.phoneme_map else (frames_dir / "phoneme_map.json")
    pmap = build_phoneme_map(tgs, map_path if map_path.exists() else None)
    if not (map_path.exists() and args.phoneme_map):
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(pmap, f, ensure_ascii=False, indent=2)
    print(f"[MAP] {map_path} (size={len(pmap)}; sil→0)")

    # 3) Upsample to 20ms phone IDs in parallel
    ok = err = 0
    work = [(p, frames_dir, pmap) for p in tgs]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_one_textgrid, w) for w in work]
        for fut in as_completed(futs):
            s, _ = fut.result()
            if s == "OK": ok += 1
            elif s == "ERR": err += 1
    print(f"[DONE] phones@20ms ids={ok} err={err} out={frames_dir}")

if __name__ == "__main__":
    main()
