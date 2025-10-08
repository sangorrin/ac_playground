# Augmentate LJSpeech_16K dataset with Voice Conversion

Here we use Coqui TTS (stable)'s FreeVC to convert an LJSpeech_16K utterance 
to the style of a VCTK speaker.

# Environment

Prepare a conda environment
```bash
conda create -n freevc -y python=3.12.3 # use same as runpod
conda activate freevc
```

Install PyTorch and dependencies
```bash
pip install torch==2.8.0 torchaudio==2.8.0 
pip install coqui-tts
pip install soundfile numpy
```

[Opt] Install fmpeg if you don't have it
```bash
brew install ffmpeg
```

# FreeVC simple 1 file test

Pick one LJSpeech_16K file and one VCTK_refs_16K reference and convert:

```bash
LJS_ID="LJ028-0386"
SPK="p264"

SRC="data/LJSpeech_16K/wavs/${LJS_ID}.wav"
REF="data/VCTK_refs_16K/${SPK}.wav"

mkdir -p data/LJS_accented_tmp data/LJS_accented
OUT_TMP="data/LJS_accented_tmp/${LJS_ID}_${SPK}.wav"
OUT_16K="data/LJS_accented/${LJS_ID}_${SPK}.wav"

tts --model_name "voice_conversion_models/multilingual/vctk/freevc24" \
    --source_wav "$SRC" \
    --target_wav "$REF" \
    --out_path "$OUT_TMP"

ffmpeg -nostdin -y -i "$OUT_TMP" -ac 1 -ar 16000 "$OUT_16K"
```

Listen
```bash
afplay "$SRC"
afplay "$REF"
afplay "$OUT_16K"
```

You should hear the **same sentence content** (from LJS) but in the **timbre/style of the VCTK speaker** you chose.

Note: the model is about 900MB and stored at /Users/dsl/Library/Application Support/tts/voice_conversion_models--multilingual--vctk--freevc24

# FreeVC batch test of 5 files

Prepare folder with the 5 LJS audios to VC.
```bash
mkdir -p data/LJSpeech_5
head -n 5 data/LJSpeech_16K/metadata.csv | cut -d'|' -f1 > tmp/ljs5.txt
while IFS= read -r ID; do
  cp "data/LJSpeech_16K/wavs/${ID}.wav" "data/LJSpeech_5/${ID}.wav"
done < tmp/ljs5.txt
```

Check you have everything ready
```bash
ls data/LJSpeech_5/*.wav | wc -l      # should be 5
ls data/VCTK_refs_16K/*.wav | head    # refs exist (one per speaker)
```

Run `freevc_batch.py` at project root
```bash
# Outputs 24 kHz VC WAVs to data/LJS_accented_tmp
python freevc_batch.py \
  --ljs_dir data/LJSpeech_5 \
  --vctk_dir data/VCTK_refs_16K \
  --out_dir data/LJS_accented_tmp \
  --num 5 \
  --workers 1
```

Resmaple to 16khz
```bash
./resample_to_16k.sh data/LJS_accented_tmp data/LJS_accented_16K
```

Quick listen
```bash
afplay data/LJSpeech_5/LJ001-0001.wav
afplay data/VCTK_refs_16K/p225.wav
afplay data/LJS_accented_16K/LJ001-0001_p225.wav
```
