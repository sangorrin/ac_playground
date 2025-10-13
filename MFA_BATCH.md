# MFA BATCH

Takes in 65k LJS accented 16k wav file and produces text files indicating the **20 ms frame → phoneme** mapping.

# Environment

Create a conda python environment
```bash
pod# sudo apt-get update -y
pod# sudo apt-get install -y libsndfile1 sox ffmpeg tree

# Miniconda
pod# cd ~
pod# wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
pod# bash miniconda.sh -b -p ~/miniconda3
pod# rm -f miniconda.sh
pod# source ~/miniconda3/etc/profile.d/conda.sh
pod# conda config --set auto_activate_base false

pod# conda create -n aligner -y python=3.11
pod# conda activate aligner

pod# conda install -c conda-forge -y montreal-forced-aligner
pod# pip install textgrid tqdm soundfile numpy

pod# mfa version
```

Download MFA english language models
```bash
pod# mfa model download dictionary english_us_mfa
pod# mfa model download acoustic english_mfa
```

# Prepare files by copying and symlinks

Generate .lab transcripts for your LJS_accented_16K (MFA needs them)
```bash
pod# python mfa_prepare.py \
      --ljs-root LJSpeech \
      --accented-wav-dir LJS_accented_16K \
      --out-corpus mfa/corpus_ljs_accented \
      --workers 32 --link hard
```

This will output on `/workspace/mfa/corpus_ljs_accented` 65k pairs like:
```
LJ001-0002_p258.lab (text/transcript of the audio)
LJ001-0002_p258.wav -> /workspace/LJS_accented_16K/LJ001-0002_p258.wav (symlink)
```

# MFA align (8 vCPUs) + 20 ms frame upsample (parallel)

```bash
# (recommended env knobs)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 4096

python mfa_upsample_batch.py \
  --corpus-dir mfa/corpus_ljs_accented \
  --dict english_us_mfa \
  --acoustic english_mfa \
  --out-align mfa/alignments_ljs_accented \
  --out-frames phones_20ms \
  --jobs 32

[Opt] rm -rf mfa/alignments_ljs_accented # temporary folder
```
