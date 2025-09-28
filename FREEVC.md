# Augmentate LJSpeech_16K dataset with Voice Conversion

Here we use Coqui TTS (stable)'s FreeVC to convert an LJSpeech_16K utterance 
to the style of a VCTK speaker.

# Environment

Prepare a conda environment
```bash
conda create -n freevc -y python=3.11
conda activate freevc
```

Install PyTorch and stable Coqui TTS
```bash
pip install torch torchvision torchaudio
pip install TTS
```

# FreeVC test

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
