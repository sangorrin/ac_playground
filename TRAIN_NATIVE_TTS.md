# TRAIN NATIVE TTS

Goal: train the coqui VITS tts with the artifacts we prepared

# Environment

In the native fs, not conda aligner env.
```bash
pod# pip install coqui-tts soundfile
```

# Check the sanity of all features

```bash
# Pad the end of the f0 frames with silence.
python scripts/fix_phones_lengths.py \
    --phones-dir phones_20ms \
    --f0-dir f0_20ms \
    --out-dir phones_20ms_fix \
    --workers 32

rm -rf phones_20ms
mv phones_20ms_fix phones_20ms

python check_features_20ms.py \
    --wav-dir LJS_accented_16K \
    --phones-dir phones_20ms \
    --f0-dir f0_20ms \
    --spk-embeds-dir VCTK_refs_16K_embeds \
    --report sanity_report.csv \
    --limit 0 \
    --workers 32
```

# Prepare manifest for coqui-tts

```bash
python make_manifest_native.py \
    --ljs-root LJSpeech \
    --wav-dir LJS_accented_16K \
    --spk-embeds-dir VCTK_refs_16K_embeds \
    --out data/manifest_coqui.jsonl \
    --workers 32
```

# Train the VITS tts using the vits_16k_ecapa.json config

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /workspace/train_jsonl.py \
  --config_path /workspace/vits_16k_ecapa.json \
  --output_path /workspace/runs/native_vits_16k
```