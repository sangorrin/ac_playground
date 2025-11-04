# Speaker embeddings

Goal: compute a fixed vector per VCTK 16k mono speaker with a **pretrained speaker encoder** (e.g., ECAPA).  

Pre-requisites: data/VCTK_refs_16K

# Execution

Prepare the environment
```bash
pod# pip install speechbrain librosa soundfile hf_transfer
```

Execute the batch (very quick)
```bash
pod# python speaker_embed_batch.py \
  --wav-dir data/VCTK_refs_16K \
  --out-dir VCTK_refs_16K_embeds \
  --batch-size 64 --device cuda
```

Output check
```bash
pod# ls VCTK_refs_16K_embeds | wc -l
pod# python
import numpy as np
print(np.load("VCTK_refs_16K_embeds/p264.npy").shape)
```