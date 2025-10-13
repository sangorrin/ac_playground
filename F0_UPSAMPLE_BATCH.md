# Frame-level F0 at 20 ms (YAAPT)

Goal: from a folder with 16k audio files produce a **20 ms frame (320 samples) → F0 (Hz)** table.
Pitch method used was YAAPT (`amfm_decompy`)

# Execution

Pre-requisites:
- folder LJS_accented_16k ready

Environment
```bash
pod#  pip install soundfile amfm_decompy
pod# python f0_20ms_batch.py \
  --in-wav-dir LJS_accented_16K \
  --out-dir f0_20ms \
  --workers 32
```

Output check
```bash
pod# ls f0_20ms | wc -l
 -> 65500 
pod# more f0_20ms/xxx
