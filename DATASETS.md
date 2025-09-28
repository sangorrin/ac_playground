# DATASETS

## LJSpeech

Download and extract the LJSpeech dataset.
```bash
mkdir -p data && curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar -xjf - -C data
mv LJSpeech-1.1 LJSpeech
```

Convert the audios to 16 Khz mono wavs.
```bash
mkdir -p data/LJSpeech_16K/wavs
find data/LJSpeech/wavs -type f -name '*.wav' | while read -r f; do
  base="$(basename "$f")"
  ffmpeg -nostdin -y -i "$f" -ac 1 -ar 16000 "data/LJSpeech_16K/wavs/$base"
done
cp data/LJSpeech/metadata.csv data/LJSpeech_16K/  # same transcripts
```

## VCTK

Download and unzip the VCTK corpus.
```bash
mkdir -p data/VCTK && curl -L https://datashare.ed.ac.uk/download/DS_10283_3443.zip -o data/DS_10283_3443.zip && unzip -q data/DS_10283_3443.zip -d data && ([ -f data/README.txt ] && mv data/README.txt data/VCTK/README_datashare.txt || true) && unzip -q data/VCTK-Corpus-0.92.zip -d data/VCTK && rm data/DS_10283_3443.zip data/VCTK-Corpus-0.92.zip
```

Prebuild a minimal 16 kHz mono refs set (one file per speaker):
```bash
mkdir -p data/VCTK_refs_16K
for d in data/VCTK/wav48_silence_trimmed/*; do
  s="$(basename "$d")"
  f="$(ls "$d"/${s}_*_mic1.flac 2>/dev/null | head -n 1)"
  [ -f "$f" ] && ffmpeg -nostdin -y -i "$f" -ac 1 -ar 16000 "data/VCTK_refs_16K/${s}.wav"
done
```

## L2-ARCTIC

1. Fill the request form: https://psi.engr.tamu.edu/l2-arctic-corpus/ (Download section).
2. In the email, choose **“L2-ARCTIC-V5.0 (everything packed)”** to get the full corpus (all 24 speakers).  
   Per-speaker links are for subsets; V1–V4 are legacy; the “Suitcase corpus” is a separate add-on not required.
3. Extract into `./data/L2-ARCTIC/`:
```bash
mkdir -p data/L2-ARCTIC && unzip -q l2arctic_release_v5.0.zip -d data/L2-ARCTIC
cd data/L2-ARCTIC && for z in *.zip; do unzip -q "$z"; done && rm -f *.zip
rm l2arctic_release_v5.0.zip
```

