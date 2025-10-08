set -e

# Pair first 10 LJS with first 10 refs (1:1)
i=0
for src in $(ls LJSpeech_10/*.wav | head -n 10); do
    i=$((i+1))
    ref=$(ls data/VCTK_refs_16K/*.wav | sed -n "${i}p")
    [ -z "$ref" ] && break
    uid=$(basename "${src%.wav}")
    spk=$(basename "${ref%.wav}")

    # FreeVC (native ~24 kHz), then resample to 16 kHz mono
    tts --use_cuda \
        --model_name "voice_conversion_models/multilingual/vctk/freevc24" \
        --source_wav "$src" \
        --target_wav "$ref" \
        --out_path "out_tmp/${uid}_${spk}.wav"

    ffmpeg -nostdin -y -i "out_tmp/${uid}_${spk}.wav" -ac 1 -ar 16000 "LJS_accented/${uid}_${spk}.wav"
done

zip -q -r LJS_accented_10.zip LJS_accented