# Run FreeVC on a Runpod GPU pod 

FreeVC's tts is quite slow (13s/audio) on Mac and we need to generate 65k audios.
Let's generate them on a cheap GPU pod from Runpod.

## Test with N files

###  Prepare the files (vctk_refs_16k.zip, ljs_16k.zip)

Prerequirements
- 16 KHz mono wavs (see DATASETS.md)
  - `data/LJSpeech_16K/wavs/*.wav` (16 kHz mono)
  - `data/VCTK_refs_16K/*.wav` (16 kHz mono, one ref per speaker)
- a Runpod with RTX 4090 (see RUNPOD.md)

Create a small N-file test set (no randomness, just first N):
```bash
rm -rf ship tmp
mkdir -p ship tmp
N=1000

# First N LJS IDs (from metadata.csv)
# [ALT] if you want ALL, just do tar zcvf ljs_16k.tar.gz data/LJSpeech_16K/wavs
# Use --no-same-owner when untarring on the server and, next time, create the tar on macOS with --no-xattrs and excludes for ._* and .DS_Store to keep the archive clean.

tar zcvf ljs_16k.tar.gz data/LJSpeech_16K/wavs

head -n "$N" data/LJSpeech/metadata.csv | cut -d'|' -f1 > "tmp/ljs.txt"
while IFS= read -r ID; do
  zip -j -q ship/ljs_16k.zip "data/LJSpeech_16K/wavs/${ID}.wav"
done < "tmp/ljs.txt"

# Zip all VCTK 16k refs (one per speaker)
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
[Alt] upload via Jupyter interface (quite useful but slow for big files)
```bash
export POD_DIR="/workspace"    # working dir on the pod
scp -pC "ship/ljs_16k.zip"         runpod-1:"$POD_DIR"/ # or ljs_16k.tar.gz
scp -pC ship/vctk_refs_16k.zip      runpod-1:"$POD_DIR"/
scp -pC freevc_batch.py            runpod-1:"$POD_DIR"/

Example: scp -rp -P 19298 -i ~/.ssh/runpod_ed25519 ljs_16k.tar.gz root@213.173.98.21:/workspace/pepito.tar.gz
```

Environment setup on pod
```bash
ssh runpod-1
pod# cd /workspace
pod# mkdir -p LJS_accented out_tmp
pod# unzip -q "ljs_16k.zip" -d "LJSpeech" # or tar xvf ljs_16k.tar.gz -C LJSpeech
pod# unzip -q vctk_refs_16k.zip
[Note] use jupyter

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
pod# pip install coqui-tts
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
--source_wav LJSpeech/LJ001-0001.wav \
--target_wav data/VCTK_refs_16K/p225.wav \
--out_path /tmp/warmup.wav
```

### Execute the batch for the 10 files

Use `screen` so that you can detach and leave the batch running
- Create session: screen -S freevc
- Detach: Ctrl + A, then D
- Reattach:	screen -r freevc

Execute the script and time it.
```bash
pod# screen -S freevc
pod# /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' python freevc_batch.py --num 1000 --workers 1 --use_cuda
pod# exit
```

[Opt] Example for the whole batch
```bash
pod# screen -S freevc
pod# /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' python freevc_batch.py --num-random-speakers 5 --workers 1 --use_cuda
pod# exit
```

Download results and terminate the pod.
[Note] for the whole 65k batch the size of out is around 20GB!
```bash
tar zcvf out.tar.gz out_tmp/
scp -pC runpod-1:"$POD_DIR"/out.tar.gz .
runpodctl stop pod $POD_ID
runpodctl remove pod $POD_ID
```

Compare the original LJSpeech wavs and the new accented ones.
```bash
tar xvf out.tar.gz
afplay data/LJSpeech_16K/wavs/LJ001-0001.wav # the original LJSpeech LJ001-0001 audio
afplay data/VCTK_refs_16K/p225.wav # VCTK speaker p225 voice
afplay out/LJ001-0001_p225.wav # LJ001-0001 but by speaker p225 from VCTK
```