# TRAIN NATIVE TTS

Goal: train the coqui VITS tts with the artifacts we prepared

# Preparation

In the native fs, not conda aligner env.
```bash
pod# pip install coqui-tts soundfile
```

<!-- [F0 NOT SUPPORTED] Pad with silence the phonemes vectors to match the length of f0 20ms vectors
```bash
python scripts/fix_phones_lengths.py \
    --phones-dir phones_20ms \
    --f0-dir f0_20ms \
    --out-dir phones_20ms_fix \
    --workers 32
rm -rf phones_20ms
mv phones_20ms_fix phones_20ms
```

[TODO UPDATE] Perform sanity checks on all the artifacts we have created so far
```bash
python check_features_20ms.py \
    --wav-dir LJS_accented_16K \
    --phones-dir phones_20ms \
    --f0-dir f0_20ms \
    --spk-embeds-dir VCTK_refs_16K_embeds \
    --report sanity_report.csv \
    --limit 0 \
    --workers 32 -->
```

Generate the manifest for coqui-tts
```bash
pod# python make_manifest_native.py \
    --ljs-root LJSpeech \
    --audio-dir LJS_accented_16K \
    --phones-dir phones_16ms \
    --dvec-dir VCTK_refs_16K_embeds \
    --output data/manifest_native.jsonl \
    --workers 32
```

# Training the native TTS

NOTE: reserve 200GB of space just in case.
NOTE: you can safe space by only conserving 1 or 3 checkpoints
config.keep_all_best=False  # Only keep best model
config.keep_after=3  # Keep last 3 checkpoints only

Train with single CPU
```bash
ulimit -n 4096
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Single RTX 5090 (default: VRAM=32 vCPU=15)
python train_native_tts.py

# [Alt] Multiple GPUs (e.g. 2xRTX 5090)
torchrun --nproc_per_node=2 train_native_tts.py

# [Alt] Other hardware (e.g., RTX 4090)
python train_native_tts.py --vram 24 --vcpus 8

# Resume training (after cancelling with Ctrl+C)
python train_native_tts.py --restore_path runs/native_vits_16k/checkpoint_25000.pth

[Alt] torchrun --nproc_per_node=2 train_native_tts.py --restore_path runs/native_vits_16k/checkpoint_25000.pth
```

# Monitoring while training

Real-time logs
```bash
tail -f runs/native_vits_16k/train.log
```

Tensorboard
```bash
tensorboard --logdir runs/native_vits_16k
# Open http://localhost:6006
# train/total_loss: Should decrease steadily
# train/mel_loss: Target ~0.3-0.35 for good quality
# eval/mel_loss: Should track train loss (no overfitting)
```

Quick quality check
```bash
# test_synthesis.py - Quick inference test
import torch
from TTS.api import TTS

# Load your checkpoint
checkpoint_path = "runs/native_vits_16k/checkpoint_15000.pth"

# Initialize with your config
model = Vits.init_from_config(config_path="runs/native_vits_16k/config.json")
model.load_checkpoint(checkpoint_path)

# Test with a sample from eval set
test_phonemes = np.load("phones_16ms/LJ001-0001_p225.npy")
test_dvec = np.load("VCTK_refs_16K_embeds/p225.npy")

# Synthesize
wav = model.inference(
    phoneme_ids=test_phonemes,
    d_vector=test_dvec,
)

# Listen
import soundfile as sf
sf.write("test_output.wav", wav, 16000)
```
