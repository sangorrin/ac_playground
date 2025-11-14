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
- Choose region where RTX 4090 availability is high (US-IL-1)
- Give it a name `ac_results_volume`
- Create a 100GB network volume to store the results longterm

GPU pod
- clikc Pods > Deploy a Pod
  - Choose: GPU, Secure Cloud, Select Network Volume, Region US-IL-1
    - Additional Filters: CUDA 12.8
  - Select RTX4090
  - Pod Template
    - Runpod Pytorch 2.8.0 (default)
    - Edit > 100 GB > Set Overrides
  - GPU Count: 1 (default)
  - On-Demand (defalt)
  - Check SSH Terminal Access and Jupyter (default)

[NOTE] The network volume gets mounted at /workspace.
If not fast enough, use the runpod SSD disk (mkdir /word)
Preparing the augmented data in /workspace is not slow and it is useful
because you can stop the pod and still have the data in the networ volume.

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
pod# git clone https://github.com/sangorrin/ac_playground.git
[Alt] scp -pC *.py  runpod-1:/workspace/
```

Install system packages
```bash
pod# apt-get update -y && apt-get install -y screen ffmpeg unzip libsndfile1 zip time sox tree
```

Install pip dependencies
```bash
pod# python -m pip install --upgrade pip
pod# pip install "huggingface_hub<1.0" soundfile amfm_decompy speechbrain librosa hf_transfer
```

Start screen session
```bash
pod# screen -S session
  # Detach: Ctrl + A, then D
  # Reattach: screen -r session
```

# 2. Download Datasets for Training

Download and extract the LJSpeech dataset at `/workspace`
```bash
pod# mkdir -p data
pod# curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar -xjf - -C data --no-same-owner
```

Download and unzip the VCTK corpus (slow, better local)
```bash
pod# mkdir -p data/VCTK && \
  curl -L https://datashare.ed.ac.uk/download/DS_10283_3443.zip -o data/DS_10283_3443.zip && \
  unzip -q data/DS_10283_3443.zip -d data && \
  unzip -q data/VCTK-Corpus-0.92.zip -d data/VCTK && \
  rm data/DS_10283_3443.zip data/VCTK-Corpus-0.92.zip
```

Create minimal VCTK refs (one 16kHz mono file per speaker)
```bash
pod# mkdir -p data/VCTK_refs_16k
pod# for d in data/VCTK/wav48_silence_trimmed/*; do
  s="$(basename "$d")"
  f="$(ls "$d"/${s}_*_mic1.flac 2>/dev/null | head -n 1)"
  [ -f "$f" ] && ffmpeg -nostdin -y -i "$f" -ac 1 -ar 16000 "data/VCTK_refs_16k/${s}.wav"
done
```

# 3. Augment Native Audio Utterances

Run FreeVC Batch (slow)
```bash
pod# mkdir -p /workspace/data/wavs_24k
pod# /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' python \
  /workspace/ac_playground/freevc_batch.py \
  --ljs_dir /workspace/data/LJSpeech-1.1/wavs \
  --vctk_dir /workspace/data/VCTK_refs_16k \
  --out_dir /workspace/data/wavs_24k \
  --num-random-speakers 5 \
  --use_cuda
```

Convert 24khz audios to 16kHz mono (fast)
```bash
pod# mkdir -p /workspace/augmented_data/wavs_16k
pod# python /workspace/ac_playground/resample_to_16k.py \
  /workspace/data/wavs_24k \
  /workspace/augmented_data/wavs_16k \
  --jobs 32 --delete-src
```

# 4. F0 Extraction (20ms frames, YAAPT)

Run F0 Batch
```bash
pod# python /workspace/ac_playground/f0_20ms_batch.py \
  --in-wav-dir /workspace/augmented_data/wavs_16k \
  --out-dir /workspace/augmented_data/f0_features \
  --workers 32
```

# 5. MFA Alignment (20ms frames)

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
pod# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
pod# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
pod# conda create -n aligner -y python=3.11
pod# conda activate aligner
pod (aligner)# conda install -c conda-forge -y montreal-forced-aligner
  [If Error] conda install -c conda-forge montreal-forced-aligner kalpy kaldi=*=cpu* openfst
pod (aligner)# pip install textgrid tqdm soundfile numpy
pod (aligner)# mfa version
```

Download MFA english models
```bash
pod (aligner)# mfa model download dictionary english_us_mfa
pod (aligner)# mfa model download acoustic english_mfa
```

Prepare Corpus files (slow)
```bash
pod (aligner)# mkdir -p /workspace/data/mfa
pod (aligner)# python /workspace/ac_playground/mfa_prepare.py \
      --ljs-root /workspace/data/LJSpeech-1.1 \
      --accented-wav-dir /workspace/augmented_data/wavs_16k \
      --out-corpus /workspace/data/mfa/corpus \
      --workers 32 --link hard
```

Run MFA + Upsample to 20ms (slow)
```bash
# Recommended env knobs
pod (aligner)# export OMP_NUM_THREADS=1
pod (aligner)# export MKL_NUM_THREADS=1
pod (aligner)# ulimit -n 4096

pod (aligner)# python /workspace/ac_playground/mfa_upsample_batch.py \
  --corpus-dir /workspace/data/mfa/corpus \
  --dict english_us_mfa \
  --acoustic english_mfa \
  --out-align /workspace/data/mfa/mfa_alignments \
  --out-frames /workspace/data/mfa/phones_20ms \
  --hop-ms 20 \
  --jobs 32
```

Pad with sil the phoneme vectors to match the length of f0 20ms vectors (fast)
```bash
pod (aligner)# python /workspace/ac_playground/fix_phones_lengths.py \
    --phones-dir /workspace/data/mfa/phones_20ms \
    --f0-dir /workspace/augmented_data/f0_features \
    --out-dir /workspace/augmented_data/mfa_alignments \
    --workers 32
pod (aligner)# conda deactivate
```

# 6. Speaker Embeddings (ECAPA-TDNN)

Run Speaker Embedding Batch
```bash
pod# python /workspace/ac_playground/speaker_embed_batch.py \
  --wav-dir /workspace/data/VCTK_refs_16k \
  --out-dir /workspace/augmented_data/speaker_embeddings \
  --batch-size 64 --device cuda
```

Verify
```bash
pod# ls /workspace/augmented_data/speaker_embeddings | wc -l
  110
pod# python
import numpy as np
print(np.load("/workspace/augmented_data/speaker_embeddings/p264.npy").shape)
  # -> (192,)
```

# 7. Sanity checks

Perform sanity checks on all the artifacts we have created so far
```bash
pod# python /workspace/ac_playground/check_features_20ms.py \
    --wav-dir /workspace/augmented_data/wavs_16k \
    --phones-dir /workspace/augmented_data/mfa_alignments \
    --f0-dir /workspace/augmented_data/f0_features \
    --spk-embeds-dir /workspace/augmented_data/speaker_embeddings \
    --report /workspace/sanity_report.csv \
    --limit 0 \
    --workers 32
[SUMMARY] total=65498 ok=65498 bad=0
[REPORT]  /workspace/sanity_report.csv
```

Get the sanity_report.csv and give it to chatGPT (if there are any errors)
```bash
mac# scp -pC runpod-1:/workspace/sanity_report.csv .
```

# 8. L2-ARCTIC Dataset for Ground Truth

Download L2-ARCTIC
1. Fill the request form: https://psi.engr.tamu.edu/l2-arctic-corpus/ (Download section).
2. In the email, choose **“L2-ARCTIC-V5.0 (everything packed)”** to get the full corpus (all 24 speakers).
[Alt] Use the copy from my drive.
```bash
pod# mkdir -p /dataset/data/ARCTIC
pod# cd /dataset/data/ARCTIC
pod# pip install gdown
pod# gdown --id 1q_Ijuz3jd3Rd2B11mB2dA_-upPeDLkxl
pod# unzip -q l2arctic_release_v5.0.zip
pod# rm l2arctic_release_v5.0.zip
pod# for z in *.zip; do unzip -q "$z"; done && rm -f *.zip
```

Resample to 16kHz
```bash
pod# mkdir -p /dataset/data/ARCTIC_16k_speakers /dataset/arctic_data/wavs_16k

# Resample each speaker folder separately (to speaker subfolders first)
pod# for speaker_dir in /dataset/data/ARCTIC/*/; do
  speaker=$(basename "$speaker_dir")
  # Skip files (LICENSE, README, etc.)
  if [[ -d "$speaker_dir" && "$speaker" != "suitcase_corpus" ]]; then
    echo "Resampling speaker: $speaker"
    python /workspace/ac_playground/resample_to_16k.py \
      "$speaker_dir/wav" \
      "/dataset/data/ARCTIC_16k_speakers/$speaker" \
      --jobs 8
  fi
done

# Flatten: Move all files to single directory with speaker suffix
pod# for speaker_dir in /dataset/data/ARCTIC_16k_speakers/*/; do
  speaker=$(basename "$speaker_dir")
  for wav_file in "$speaker_dir"/*.wav; do
    if [[ -f "$wav_file" ]]; then
      basename=$(basename "$wav_file" .wav)
      mv "$wav_file" "/dataset/arctic_data/wavs_16k/${basename}_${speaker}.wav"
    fi
  done
done

# Cleanup storage
pod# rm -rf /dataset/data/ARCTIC_16k_speakers
```

Extract F0 (20ms frames)
```bash
pod# python /workspace/ac_playground/f0_20ms_batch.py \
  --in-wav-dir /dataset/arctic_data/wavs_16k \
  --out-dir /dataset/arctic_data/f0_features \
  --workers 32
```

MFA Alignment (20ms frames)
```bash
pod# conda activate aligner

# Prepare MFA corpus (creates .wav + .lab files from ARCTIC transcripts)
pod (aligner)# python /workspace/ac_playground/mfa_prepare.py \
  --arctic-root /dataset/data/ARCTIC \
  --wav-dir /dataset/arctic_data/wavs_16k \
  --out-corpus /dataset/arctic_data/mfa_corpus \
  --workers 8 --link hard

# Run MFA + upsample to 20ms
pod (aligner)# export OMP_NUM_THREADS=1
pod (aligner)# export MKL_NUM_THREADS=1
pod (aligner)# ulimit -n 4096

pod (aligner)# python /workspace/ac_playground/mfa_upsample_batch.py \
  --corpus-dir /dataset/arctic_data/mfa_corpus \
  --dict english_us_mfa \
  --acoustic english_mfa \
  --out-align /dataset/arctic_data/mfa_alignments_raw \
  --out-frames /dataset/arctic_data/mfa_phones_20ms \
  --hop-ms 20 \
  --jobs 24 # only 24 speakers

# Fix lengths to match F0
pod (aligner)# python /workspace/ac_playground/fix_phones_lengths.py \
  --phones-dir /dataset/arctic_data/mfa_phones_20ms \
  --f0-dir /dataset/arctic_data/f0_features \
  --out-dir /dataset/arctic_data/mfa_alignments \
  --workers 24
pod (aligner)# conda deactivate
```

Extract one embedding per ARCTIC speaker
```bash
pod# python /workspace/ac_playground/speaker_embed_batch.py \
  --wav-dir /dataset/arctic_data/wavs_16k \
  --out-dir /dataset/arctic_data/speaker_embeddings \
  --batch-size 64 \
  --device cuda
```

Verify L2-ARCTIC features.
```bash
pod# python /workspace/ac_playground/check_features_20ms.py \
  --wav-dir /dataset/arctic_data/wavs_16k \
  --phones-dir /dataset/arctic_data/mfa_alignments \
  --f0-dir /dataset/arctic_data/f0_features \
  --spk-embeds-dir /dataset/arctic_data/speaker_embeddings \
  --report arctic_sanity_report.csv \
  --workers 32
```

# Estimated times and resource usage

Time: 7h
Network volume: 80GB (only 13GB are from augmented_data)
Pod disk: 15GB

```bash
pod# du -h augmented_data/
  1.1M	augmented_data/speaker_embeddings
  67M	augmented_data/mfa_alignments
  108M	augmented_data/f0_features
  13G	augmented_data/wavs_16k
  13G	augmented_data/
```

