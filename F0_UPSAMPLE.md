# Frame-level F0 at 20 ms (YAAPT)

Goal: from an audio file produce a **20 ms frame → F0 (Hz)** table that matches the MFA 20 ms grid.

- **Text (for pairing later):** `data/LJSpeech/metadata.csv` (LJS transcript of the chosen ID)
- **Audio (for F0 here):** `data/LJSpeech/wavs/<ID>.wav` or `data/LJS_accented/<ID>_<SPK>.wav`
- **Pitch method:** YAAPT (`amfm_decompy`)
- **Time base:** 16 kHz audio, hop = **320 samples** = **20 ms**

# Example

Environment
```bash
conda create -n f0 -y python=3.11
conda activate f0
pip install "numpy<2.3" "scipy==1.14.1" soundfile amfm_decompy
```

Choose an utterance
```bash
ID="LJ028-0386"
WAV="$(pwd)/data/LJSpeech/wavs/${ID}.wav"   # or data/LJS_accented/${ID}_<SPK>.wav
echo "WAV=$WAV"
```

Run the extractor (prints to STDOUT)
```bash
python f0_20ms.py "$WAV"
```

`f0_20ms.py` does:
- Load audio, convert to **mono / 16 kHz**
- Run **YAAPT** with **20 ms** frame spacing
- Print: `frame_idx  start_sec  end_sec  F0_Hz  VUV`

Example output (first lines)
```
frame_idx  start_sec  end_sec  F0_Hz  VUV
0          0.000      0.020    0.00   0
1          0.020      0.040    118.4  1
2          0.040      0.060    120.1  1
...
```

- **F0_Hz = 0.00** when **unvoiced** (VUV = 0).
- Row count ≈ duration / **0.02 s** and should match the MFA 20 ms phoneme table for the same utterance.
- Keep this F0 env (`f0`) separate from the MFA env (`aligner`) to avoid dependency conflicts.
- If you switch to a FreeVC file, keep the same **LJS ID** so it pairs with the LJS text later.
- To save to a file: `python f0_20ms.py "$WAV" > out/f0_${ID}.tsv`
