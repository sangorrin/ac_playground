# Run FreeVC on a Runpod GPU pod 

FreeVC's tts is quite slow (13s/audio) on Mac and we need to generate 65k audios.
Let's generate them on a cheap GPU pod from Runpod.

## Test with 10 files

###  Prepare the 10 files (vctk_refs_16k.zip, ljs_10_16k.zip)

Prerequirements
- 16 KHz mono wavs (see DATASETS.md)
  - `data/LJSpeech_16K/wavs/*.wav` (16 kHz mono)
  - `data/VCTK_refs_16K/*.wav` (16 kHz mono, one ref per speaker)
- a Runpod with RTX 4090 (see RUNPOD.md)

Create a small 10-file test set (no randomness, just first 10):
```bash
mkdir -p ship tmp

# First 10 LJS IDs (from metadata.csv)
head -n 10 data/LJSpeech/metadata.csv | cut -d'|' -f1 > tmp/ljs10.txt

# Zip those 10 LJS 16k WAVs
rm -f ship/ljs_10_16k.zip
while read -r ID; do
    zip -j -q ship/ljs_10_16k.zip "data/LJSpeech_16K/wavs/${ID}.wav"
done < tmp/ljs10.txt

# Zip all VCTK 16k refs (one per speaker)
rm -f ship/vctk_refs_16k.zip
zip -r -q ship/vctk_refs_16k.zip data/VCTK_refs_16K
```

### [ONCE] Pod setup

[Once] creaet your pod as in RUNPOD.md
[Opt] Start your pod if it was stopped
```bash
export POD_ID="YOUR_POD_ID"  # runpodctl get pod
runpodctl get pods
runpodctl start pod $POD_ID
```

Create the pod folder and upload the two zips and script via SCP:
[Alt] upload via Jupyter interface (quite useful)
```bash
export POD_DIR="/workspace"    # working dir on the pod
scp -pC ship/ljs_10_16k.zip         runpod-1:"$POD_DIR"/
scp -pC ship/vctk_refs_16k.zip      runpod-1:"$POD_DIR"/
scp -pC freevc_runpod.sh            runpod-1:"$POD_DIR"/
```

Environment setup on pod
```bash
ssh runpod-1
pod# cd /workspace
pod# mkdir -p LJS_accented out_tmp
pod# unzip -q ljs_10_16k.zip -d LJSpeech_10
pod# unzip -q vctk_refs_16k.zip
[Note] use jupyter

pod# set -e
pod# apt-get update -y && apt-get install -y ffmpeg unzip libsndfile1 zip time

# Fresh python sandbox/env. 
pod# python3 -m venv --system-site-packages ~/envs/freevc
pod# source ~/envs/freevc/bin/activate
pod# python -V
  -> 3.12.3

# Check CUDA and torch is available.
pod# python -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'cuda', torch.cuda.is_available())"
  -> Output: torch 2.8.0+cu128 torchaudio 2.8.0+cu128 cuda True

# Install coqui-tts (maintained fork; keeps the tts CLI).
pod# python -m pip install --upgrade pip
pod# pip install --no-cache-dir "coqui-tts==0.27.*"
```

Check Torch sees the GPU
```bash
pod# python
import torch
print("CUDA available:", torch.cuda.is_available())
  -> CUDA available: True
print("Device count:", torch.cuda.device_count())
  -> Device count: 1
print("Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
  -> Name: NVIDIA GeForce RTX 4090
```

Load FreeVC model once to warm the cache (downloads to ~/.local / HF cache)
```bash
pod# python
from TTS.api import TTS
tts = TTS("voice_conversion_models/multilingual/vctk/freevc24").to("cuda")
print("FreeVC loaded on:", next(tts.voice_converter.parameters()).device)
  -> FreeVC loaded on: cuda:0
```

CLI warm-up with a single file (ensures the speaker encoder also runs on GPU)
```bash
pod# tts --use_cuda \
--model_name "voice_conversion_models/multilingual/vctk/freevc24" \
--source_wav LJSpeech_10/LJ001-0001.wav \
--target_wav data/VCTK_refs_16K/p225.wav \
--out_path /tmp/warmup.wav
```

### Execute the batch for the 10 files

Execute the script and time it.
```bash
pod# /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' bash freevc_runpod.sh
  -> elapsed=3:16.45 cpu=198% maxrss=3834880KB
```

Download results and terminate the pod.
[Alt] Use Jupyter to download and the UI to stop/terminate the pod.
```bash
scp -pC runpod-1:"$POD_DIR"/LJS_accented_10.zip .
runpodctl stop pod $POD_ID
runpodctl remove pod $POD_ID
```

Compare the original LJSpeech wavs and the new accented ones.
```bash
afplay data/LJSpeech_16K/wavs/LJ001-0001.wav # the original LJSpeech LJ001-0001 audio
afplay data/VCTK_refs_16K/p225.wav # VCTK speaker p225 voice
afplay LJS_accented/LJ001-0001_p225.wav # LJ001-0001 but by speaker p225 from VCTK
```