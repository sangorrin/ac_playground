# MFA_UPSAMPLE.md — Frame-level phonemes from LJSpeech text (via MFA)

Goal: from LJSpeech text (source: data/LJSpeech/metadata.csv) and an audio of the same LJS sentence, produce a 20 ms frame → phoneme table.  
Phoneme set comes from MFA dictionary: english_us_mfa.

# Example

Choose the utterance
```bash
# Reuse MFA.md environment and models.
conda activate aligner

# Pick an ID present in LJSpeech
ID="LJ028-0386"

# Pick a WAV from LJSpeech
# Note: You can also use a FreeVC-generated audio for the same ID/text
WAV="$(pwd)/data/LJSpeech/wavs/${ID}.wav"

# Text FROM data/LJSpeech/metadata.csv
# column 1 = ID, column 2 = normalized text, column 3 = raw text (fallback)
TEXT="$(grep "^${ID}|" data/LJSpeech/metadata.csv | cut -d'|' -f3)"

echo "ID=$ID"
echo "WAV=$WAV"
echo "TEXT=$TEXT"
```

Run MFA align to get a TextGrid (contains phone intervals).
```bash
rm -rf work && mkdir -p work/corpus work/aligned
cp "$WAV" "work/corpus/"
echo -n "$TEXT" > "work/corpus/${ID}.lab"

mfa align --clean -j 1 work/corpus english_us_mfa english_mfa work/aligned
less "work/aligned/${ID}.TextGrid"
```

Upsample to 20 ms frames (print frame → phoneme on STDOUT)
```bash
TG="work/aligned/${ID}.TextGrid"
python mfa_upsample.py "$TG"
```

Example output (first lines):

    frame_idx  start_sec  end_sec  PHONEME
    0          0.000      0.020    sil
    1          0.020      0.040    DH
    2          0.040      0.060    AH0
    3          0.060      0.080    S
    ...

- PHONEME labels come from MFA’s english_us_mfa dictionary.
- Number of rows ≈ duration / 0.02 s.
- Using FreeVC audio for the same ID: set WAV to data/LJS_accented/${ID}_<SPK>.wav, keep TEXT from metadata.csv
