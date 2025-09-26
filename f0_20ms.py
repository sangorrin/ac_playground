# f0_20ms.py — YAAPT F0 at 20 ms frames (robust, no samp_times)
# Usage: python f0_20ms.py path/to/audio.wav
import sys, os, math, tempfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from amfm_decompy.basic_tools import SignalObj
from amfm_decompy.pYAAPT import yaapt

FS_TARGET = 16000
HOP_SEC   = 0.020  # 20 ms

def to_tmp_16k(path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != FS_TARGET:
        x = resample_poly(x, FS_TARGET, sr)
        sr = FS_TARGET
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, x, sr, subtype="PCM_16")
    tmp.close()
    return tmp.name, sr, len(x)

def main():
    if len(sys.argv) < 2:
        print("Usage: python f0_20ms.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    in_wav = sys.argv[1]
    tmp_wav, sr, n_samps = to_tmp_16k(in_wav)
    dur = n_samps / sr
    try:
        sig = SignalObj(tmp_wav)  # this version expects a filename
        pitch = yaapt(
            sig,
            frame_length=25.0,            # ms
            frame_space=HOP_SEC * 1000,   # ms
            f0_min=60.0,
            f0_max=400.0,
            otime_threshold=0.0,
        )

        # Robust field access across amfm_decompy versions
        f0_vals = np.asarray(getattr(pitch, "samp_values", getattr(pitch, "f0", [])), dtype=float)
        vuv     = np.asarray(getattr(pitch, "vuv", np.zeros_like(f0_vals)), dtype=int)

        # Build exact 20 ms frame grid
        n_frames = int(math.ceil(dur / HOP_SEC))
        n_f0 = len(f0_vals)

        print("frame_idx\tstart_sec\tend_sec\tF0_Hz\tVUV")
        for i in range(n_frames):
            t0 = i * HOP_SEC
            t1 = min((i + 1) * HOP_SEC, dur)
            j = i if n_f0 else -1  # index-aligned grid; fallback to -1 if empty
            f0 = float(f0_vals[j]) if 0 <= j < n_f0 else 0.0
            vu = int(vuv[j])       if 0 <= j < n_f0 else 0
            if not np.isfinite(f0) or f0 < 0 or vu == 0:
                f0 = 0.0
            print(f"{i}\t{t0:.3f}\t{t1:.3f}\t{f0:.2f}\t{vu}")
    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass

if __name__ == "__main__":
    main()
