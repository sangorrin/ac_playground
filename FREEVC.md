# Augmentate LJSpeech dataset with Voice Conversion

Here we use Coqui TTS (stable)'s FreeVC to convert an LJSpeech utterance 
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

Pick one LJSpeech file and one VCTK reference and convert:

```bash
LJS_ID="LJ028-0386"
SPK="p264"

SRC="data/LJSpeech/wavs/${LJS_ID}.wav"
REF="$(ls data/VCTK/wav48_silence_trimmed/${SPK}/${SPK}_*_mic1.flac | head -n 1)"

mkdir -p data/LJS_accented
OUT="data/LJS_accented/${LJS_ID}_${SPK}.wav"

tts --model_name "voice_conversion_models/multilingual/vctk/freevc24" \
    --source_wav "$SRC" \
    --target_wav "$REF" \
    --out_path "$OUT"
```

Listen
```bash
afplay "$SRC"
afplay "$REF"
afplay "$OUT"
```

You should hear the **same sentence content** (from LJS) but in the **timbre/style of the VCTK speaker** you chose.

Note: the model is about 900MB and stored at /Users/dsl/Library/Application Support/tts/voice_conversion_models--multilingual--vctk--freevc24
