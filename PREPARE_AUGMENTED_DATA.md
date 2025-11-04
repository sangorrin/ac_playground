# PREPARE AUGMENTED DATA

This guide describes the complete workflow for preparing the augmented dataset for Native TTS training, as described in https://arxiv.org/abs/2506.16580 using a modified Coqui's VITS model to support Native TTS

Target Dataset Format
```
/workspace/augmented_dataset/
  ├── wavs_16k/LJ001-0001_p225.wav (16kHz, ljspeech_id + vctk_speaker via freevc)
  ├── mfa_alignments/LJ001-0001_p225.npy (aligned phoneme IDs, 20ms frames, int16)
  ├── f0_features/LJ001-0001_p225.npy (F0 values, 20ms frames, YAAPTS)
  └── speaker_embeddings/p225.npy (192-dim ECAPA embedding per VCTK speaker)
```

# 1. RunPod Setup

## [ONCE] Generate a new key pair (ed25519)

Create the private and public key, and upload the public key to runpod.
```bash
ssh-keygen -t ed25519 -a 100 -C "runpod" -f ~/.ssh/runpod_ed25519
chmod 600 ~/.ssh/runpod_ed25519

# Copy the public key and paste it in RunPod Settings → SSH Public Keys → Update
cat ~/.ssh/runpod_ed25519.pub
```

## Deploy a Runpod

Network volume
- Create a 100GB network volume to store the results longterm

GPU pod
- clikc Pods > Deploy a Pod
  - Choose: GPU, Secure Cloud, Any Region
    [Note] Do not choose Community Cloud (network problems)
  - Additional Filters:
    - select same CUDA version as template (e.g. 12.8)
  - Select RTX4090
  - Storage: set the size to 100GB (you can edit the pod to increase it)
  - Attach a Network Volume storage with 100GB
  - Select Pod Template
    - Runpod Pytorch 2.8.0
    - runpod/pytorch:1.0.1-cu1281-torch280-ubuntu2404
  - Select On-Demand
  - Check SSH Terminal Access and Jupyter

## Accessing the POD via SSH

Start the pod and get the SSH IP/PORT from “Connect” tab.
Then fill the ssh config so you can access by SSH easily.
[Note] If you stop/restart the pod, the IP/PORT will change.

```bash
cat >> ~/.ssh/config <<'EOF'
Host runpod-1
  HostName <POD_IP>
  Port <PORT>
  User root
  IdentityFile ~/.ssh/runpod_ed25519
  IdentitiesOnly yes
EOF

ssh runpod-1
pod# screen -S session
  # Detach: Ctrl + A, then D
  # Reattach: screen -r session
```

Check CUDA and torch
```bash
pod# python
import torch
print('torch', torch.__version__)
  # -> torch 2.8.0+cu128 torchaudio 2.8.0+cu128 cuda True
print("CUDA available:", torch.cuda.is_available())
  # -> CUDA available: True
print("Device count:", torch.cuda.device_count())
  # -> Device count: 1
print("Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
  # -> Name: NVIDIA GeForce RTX 4090
```

## Dependencies

Get the scripts
```bash
pod# cd /workspace
pod# git clone ...
[Alt] scp -pC *.py  runpod-1:/workspace/
```

Install system packages
```bash
pod# apt-get update -y && apt-get install -y ffmpeg unzip libsndfile1 zip time sox tree
```

Install pip dependencies
```bash
pod# python -m pip install --upgrade pip
pod# pip install coqui-tts soundfile amfm_decompy speechbrain librosa hf_transfer
```

# 2. Download Datasets

Download and extract the LJSpeech dataset.
```bash
pod# mkdir -p data
pod# curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar -xjf - -C data
```

Download and unzip the VCTK corpus.
```bash
pod# mkdir -p data/VCTK && \
  curl -L https://datashare.ed.ac.uk/download/DS_10283_3443.zip -o data/DS_10283_3443.zip && \
  unzip -q data/DS_10283_3443.zip -d data && \
  unzip -q data/VCTK-Corpus-0.92.zip -d data/VCTK && \
  rm data/DS_10283_3443.zip data/VCTK-Corpus-0.92.zip
```

Create minimal VCTK refs (one 16kHz mono file per speaker)
```bash
mkdir -p data/VCTK_refs_16K
for d in data/VCTK/wav48_silence_trimmed/*; do
  s="$(basename "$d")"
  f="$(ls "$d"/${s}_*_mic1.flac 2>/dev/null | head -n 1)"
  [ -f "$f" ] && ffmpeg -nostdin -y -i "$f" -ac 1 -ar 16000 "data/VCTK_refs_16K/${s}.wav"
done
```

# 3. Augment Native Audio Utterances

Run FreeVC Batch
```bash
pod# mkdir -p data/out_24k
pod# /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' python freevc_batch.py \
  --ljs_dir data/LJSpeech-1.1/wavs \
  --vctk_dir data/VCTK/wav48_silence_trimmed \
  --out_dir data/out_24k \
  --num-random-speakers 5 \
  --use_cuda
```

Convert 24khz audios to 16kHz mono
```bash
pod# mkdir -p /workspace/augmented_data/wavs_16k
pod# python resample_to_16k.py \
  /workspace/data/out_24k \
  /workspace/augmented_data/wavs_16k \
  --jobs 32 --delete-src
```

# 4. MFA Alignment (20ms frames)

We need to run this on a conda environment for compatibility
```bash
pod# cd ~
pod# wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
pod# bash miniconda.sh -b -p ~/miniconda3
pod# rm -f miniconda.sh
pod# source ~/miniconda3/etc/profile.d/conda.sh
pod# conda config --set auto_activate_base false
```

Create aligner environment
```bash
pod# conda create -n aligner -y python=3.11
pod# conda activate aligner
pod# conda install -c conda-forge -y montreal-forced-aligner
pod# pip install textgrid tqdm soundfile numpy
pod# mfa version
```

Download MFA english models
```bash
pod# mfa model download dictionary english_us_mfa
pod# mfa model download acoustic english_mfa
```

Prepare Corpus files (`/workspace/mfa/corpus/{speaker}/{utt_id}_{speaker}.{wav,lab}`)
```bash
pod# mkdir -p mfa
pod# python mfa_prepare.py \
      --ljs-root data/LJSpeech-1.1
      --accented-wav-dir augmented_data/wavs_16k \
      --out-corpus mfa/corpus \
      --workers 32 --link hard
```

Run MFA + Upsample to 20ms
```bash
# Recommended env knobs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 4096

pod# python mfa_upsample_batch.py \
  --corpus-dir mfa/corpus \
  --dict english_us_mfa \
  --acoustic english_mfa \
  --out-align mfa/alignments_20ms \
  --out-frames /workspace/augmented_data/mfa_alignments \
  --hop-ms 20 \
  --jobs 32

pod# rm -rf mfa/
```

# 5. F0 Extraction (20ms frames, YAAPT)

Run F0 Batch
```bash
pod# python f0_20ms_batch.py \
  --in-wav-dir /workspace/augmented_data/wavs_16k \
  --out-dir /workspace/augmented_data/f0_features \
  --workers 32
```

## 6. Speaker Embeddings (ECAPA-TDNN)

Run Speaker Embedding Batch
```bash
pod# python speaker_embed_batch.py \
  --wav-dir data/VCTK_refs_16K \
  --out-dir /workspace/augmented_data/speaker_embeddings \
  --batch-size 64 --device cuda
```

Verify
```bash
pod# ls VCTK_refs_16K_embeds | wc -l
pod# python
import numpy as np
print(np.load("VCTK_refs_16K_embeds/p264.npy").shape)
  # -> (192,)
```
