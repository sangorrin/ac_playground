# Train Standard Vits

Goal: get some practice training and infering with the framework coqui-tts.

# Environment

Runpod with RTX4090
```bash
pod# apt-get update -y && apt-get install -y unzip libsndfile1 zip time screen
pod# python -m pip install --upgrade pip
pod# pip install tensorboard soundfile matplotlib

pod# cd /workspace
pod# git clone https://github.com/idiap/coqui-ai-TTS.git
pod# cd coqui-ai-TTS
pod# pip install -e .
```

# Download dataset

Download the ljspeech dataset in the appropriate folder
```bash
pod# python -c "from TTS.utils.downloaders import download_ljspeech; download_ljspeech('recipes/ljspeech/LJSpeech-1.1')"

TODO: fix recipes/ljspeech/LJSpeech-1.1/LJSpeech-1.1
```

This will create:
```
recipes/ljspeech/LJSpeech-1.1/
  ├── metadata.csv
  └── wavs/
```

That matches the hardcoded expectation in:
```python
path=os.path.join(output_path, "../LJSpeech-1.1/")
```

# (Optional) Adjust training hyperparameters

Edit `recipes/ljspeech/vits_tts/train_vits.py` and modify:

```python
batch_size=64,             # You can increase to 64 or 96 on RTX 4090
epochs=500,               # Reduce to 300 or 500 if you just want a quick test
```

# Train directly

```bash
pod# screen -S vits
pod# cd recipes/ljspeech/vits_tts

[Optional]
pod# ulimit -n 4096
pod# export OMP_NUM_THREADS=1
pod# export MKL_NUM_THREADS=1
pod# export OPENBLAS_NUM_THREADS=1

pod# python train_vits.py
    detach: Ctrl + A, then D
pod# screen -r vits
```

It will:
* Automatically locate `../LJSpeech-1.1/`
* Create checkpoints and logs in `recipes/ljspeech/vits_tts/vits_ljspeech/`

# Monitor training

```bash
tensorboard --logdir recipes/ljspeech/vits_tts/vits_ljspeech
```

# Generate audio (after training)

Once a checkpoint is available (e.g., `checkpoint_xxx.pth`):

```bash
python TTS/bin/tts_infer.py \
  --model_path recipes/ljspeech/vits_tts/vits_ljspeech/checkpoint_xxx.pth \
  --text "The quick brown fox jumps over the lazy dog."
```

The WAV will appear in the same output folder.