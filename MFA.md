# MFA (Monstreal Forced Alignement)

Takes in a wav file and produces a text file indicating the **20 ms frame → phoneme** mapping. 

## Environment

Create a conda python environment
```bash
conda create -n aligner -y python=3.11
conda activate aligner
conda install -c conda-forge -y montreal-forced-aligner
pip install textgrid
```

Download MFA english language models
```bash
mfa model download dictionary english_us_mfa
mfa model download acoustic english_mfa
tree ~/Documents/MFA
```

## Example

Choose a random audio file and text
```bash
# L2-ARCTIC
WAV="$(pwd)/data/L2-ARCTIC/ABA/wav/arctic_a0179.wav"
TEXT="$(cat data/L2-ARCTIC/ABA/transcript/arctic_a0179.txt)"

# VCTK
# WAV="$(pwd)/data/VCTK/wav48_silence_trimmed/p264/p264_264_mic1.flac"
# TEXT="$(cat data/VCTK/txt/p264/p264_264.txt)"

# LJSpeech
# WAV="$(pwd)/data/LJSpeech-1.1/wavs/LJ028-0386.wav"
# TEXT="$(cat data/LJSpeech-1.1/metadata.csv | grep '^LJ028-0386|')"
```

Call MFA align
```bash
rm -rf work
mkdir -p work/corpus work/aligned

U="$(basename "$WAV")"; U="${U%.*}"
cp "$WAV" work/corpus/
printf '%s\n' "$TEXT" > "work/corpus/$U.lab"

mfa align --clean -j 1 work/corpus english_us_mfa english_mfa work/aligned
less "work/aligned/${U}.TextGrid"
```
