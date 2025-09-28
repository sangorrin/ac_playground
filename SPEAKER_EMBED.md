# Speaker embeddings

Goal: compute a fixed vector with a **pretrained speaker encoder** (e.g., ECAPA).  

## Environment

```bash
conda create -n spk -y python=3.11
conda activate spk

# CPU torch + torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Speaker encoder (ECAPA), IO utils
pip install speechbrain librosa soundfile numpy
```

## Example: one embedding from a single reference utterance

Choose a **canonical** file per speaker (e.g., the first clean `mic1.flac`).
```bash
mkdir -p data/spk_embed/ecapa
python speaker_embed.py --speaker p264 --reference data/VCTK/wav48_silence_trimmed/p264/p264_001_mic1.flac
ls data/spk_embed/ecapa/p264.npy   # ~192-dim vector
```

Check one vector’s shape (should be 192,)
```python
import numpy as np
print(np.load("data/spk_embed/ecapa/p264.npy").shape)
```

## Bulk embeddings for many VCTK speakers

```bash
python speaker_embed.py --batch
ls data/spk_embed/ecapa/*.npy | wc -l
```