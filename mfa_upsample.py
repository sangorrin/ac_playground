# mfa_upsample.py (fixed)
import sys, math
from textgrid import TextGrid, IntervalTier

H = 0.020  # 20 ms
tg = TextGrid.fromFile(sys.argv[1])

# 1) pick the 'phones' tier if present; else the largest IntervalTier
tier = None
for t in tg.tiers:
    name = (getattr(t, "name", "") or "").strip().lower()
    if isinstance(t, IntervalTier) and name in {"phones", "phone", "phoneme"}:
        tier = t
        break

if tier is None:
    itiers = [t for t in tg.tiers if isinstance(t, IntervalTier)]
    if not itiers:
        raise RuntimeError("No IntervalTier found in TextGrid.")
    tier = max(itiers, key=lambda x: len(getattr(x, "intervals", [])))

# 2) upsample to 20 ms frames
t_end = max([i.maxTime for i in tier.intervals] + [getattr(tg, "maxTime", 0.0)])
n = int(math.ceil(t_end / H))

print("frame_idx\tstart_sec\tend_sec\tPHONEME")
for i in range(n):
    t0 = i * H
    t1 = min((i + 1) * H, t_end)
    tc = 0.5 * (t0 + t1)
    ph = "sil"
    for itv in tier.intervals:
        if itv.minTime <= tc < itv.maxTime:
            mark = (itv.mark or "").strip()
            ph = mark if mark else "sil"
            break
    print(f"{i}\t{t0:.3f}\t{t1:.3f}\t{ph}")
