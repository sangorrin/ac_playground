#!/usr/bin/env python3
"""
Native TTS Training Script (Stage 1) - FINAL VERSION
Trains VITS with pre-computed phoneme alignments from MFA at 16ms frames

Validated against: coqui-ai-TTS v0.27.2 and coqui-ai-Trainer v0.3.1
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseDatasetConfig

# ============================================================================
# DATASET - Matching VitsDataset output format exactly
# ============================================================================

class NativeTTSDataset(Dataset):
    """Dataset that loads pre-computed phonemes and outputs VITS-compatible batch format"""

    def __init__(self, samples, ap, d_vector_dim=192):
        self.samples = samples
        self.ap = ap
        self.d_vector_dim = d_vector_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load audio (returns wav array, not tuple)
        wav = self.ap.load_wav(item["audio_file"])

        # Load pre-computed phoneme IDs
        phoneme_ids = np.load(item["phoneme_file"]).astype(np.int64)

        # Load d-vector
        d_vector = np.load(item["d_vector_file"]).astype(np.float32)

        return {
            "raw_text": item["text"],
            "token_ids": phoneme_ids,
            "token_len": len(phoneme_ids),
            "wav": wav,
            "wav_len": len(wav),
            "speaker_name": item["speaker_name"],
            "d_vector": d_vector,
            "audio_file": item["audio_file"],
            "audio_unique_name": Path(item["audio_file"]).stem,
            "language_name": "native",
        }


def collate_fn(batch):
    """
    Collate function matching EXACTLY the VitsDataset.collate_fn output format
    Based on TTS/tts/models/vits.py VitsDataset.collate_fn
    """
    # Find max lengths
    max_token_len = max([item["token_len"] for item in batch])
    max_wav_len = max([item["wav_len"] for item in batch])

    # Initialize tensors
    batch_size = len(batch)
    token_padded = torch.zeros(batch_size, max_token_len, dtype=torch.long)
    token_lens = torch.zeros(batch_size, dtype=torch.long)
    token_rel_lens = torch.zeros(batch_size, dtype=torch.float)

    # CRITICAL: waveform must be 3D [B, 1, T]
    wav_padded = torch.zeros(batch_size, 1, max_wav_len, dtype=torch.float)
    wav_lens = torch.zeros(batch_size, dtype=torch.long)
    wav_rel_lens = torch.zeros(batch_size, dtype=torch.float)

    d_vectors = torch.stack([torch.FloatTensor(item["d_vector"]) for item in batch])

    # Lists for metadata
    speaker_names = []
    language_names = []
    audio_files = []
    raw_texts = []
    audio_unique_names = []

    # Fill tensors
    for i, item in enumerate(batch):
        # Tokens
        token_padded[i, :item["token_len"]] = torch.LongTensor(item["token_ids"])
        token_lens[i] = item["token_len"]
        token_rel_lens[i] = item["token_len"] / max_token_len

        # Waveform (add channel dimension)
        wav_padded[i, 0, :item["wav_len"]] = torch.FloatTensor(item["wav"])
        wav_lens[i] = item["wav_len"]
        wav_rel_lens[i] = item["wav_len"] / max_wav_len

        # Metadata
        speaker_names.append(item["speaker_name"])
        language_names.append(item["language_name"])
        audio_files.append(item["audio_file"])
        raw_texts.append(item["raw_text"])
        audio_unique_names.append(item["audio_unique_name"])

    # Return batch matching EXACT VitsDataset format
    # CRITICAL: Key is "tokens" not "token_ids" for train_step
    return {
        "tokens": token_padded,
        "token_lens": token_lens,
        "token_rel_lens": token_rel_lens,
        "waveform": wav_padded,  # [B, 1, T] - 3D!
        "waveform_lens": wav_lens,
        "waveform_rel_lens": wav_rel_lens,
        "speaker_names": speaker_names,
        "language_names": language_names,
        "audio_files": audio_files,
        "raw_text": raw_texts,
        "audio_unique_names": audio_unique_names,
        "d_vectors": d_vectors,
    }


# ============================================================================
# FORMATTER
# ============================================================================

def native_tts_formatter(root_path, meta_file, **kwargs):
    """Load manifest with pre-computed files"""
    items = []
    manifest_path = Path(root_path) / meta_file

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            items.append({
                "text": data["text"],
                "audio_file": str(Path(root_path) / data["audio_file"]),
                "phoneme_file": str(Path(root_path) / data["phoneme_file"]),
                "d_vector_file": str(Path(root_path) / data["d_vector_file"]),
                "speaker_name": data["speaker_name"],
            })

    print(f"[Formatter] Loaded {len(items)} samples")
    return items


# ============================================================================
# PRE-TRAINED MODEL
# ============================================================================

def download_pretrained_model():
    """Download pre-trained VITS model"""
    from TTS.utils.manage import ModelManager

    print("\n" + "="*80)
    print("📥 Downloading pre-trained VITS model for transfer learning...")
    print("   Model: tts_models/eng/fairseq/vits (16kHz)")
    print("="*80 + "\n")

    manager = ModelManager()
    model_path, _, _ = manager.download_model("tts_models/eng/fairseq/vits")
    model_file = Path(model_path) / "model.pth"

    print(f"✅ Downloaded: {model_file}\n")
    return str(model_file)


def load_pretrained_weights(model, pretrained_path):
    """Load compatible weights from pre-trained model"""
    print("\n" + "="*80)
    print(f"🔄 Loading pre-trained weights from:")
    print(f"   {pretrained_path}")
    print("="*80 + "\n")

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    pretrained_dict = checkpoint.get("model", checkpoint)
    model_dict = model.state_dict()

    loaded = []
    skipped = []

    for name, param in pretrained_dict.items():
        if name in model_dict and model_dict[name].shape == param.shape:
            model_dict[name] = param
            loaded.append(name)
        else:
            skipped.append(name)

    model.load_state_dict(model_dict, strict=False)

    print(f"✅ Loaded {len(loaded)}/{len(pretrained_dict)} layers")
    if skipped:
        print(f"   Skipped {len(skipped)} incompatible layers (text encoder size mismatch)")
    print("="*80 + "\n")

    return model


# ============================================================================
# HARDWARE CONFIG
# ============================================================================

def get_hardware_config(vram_gb=None, vcpus=None):
    """Calculate optimal training parameters"""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if vram_gb is None and num_gpus > 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if vcpus is None:
        vcpus = os.cpu_count() or 8

    # Calculate batch size (~0.9 GB per sample for VITS 16kHz)
    if vram_gb:
        batch_size = max(8, int((vram_gb - 2) * 1.1))
    else:
        batch_size = 16

    batch_size = min(batch_size, 32)
    num_workers = max(4, min(vcpus - 2, 8))

    return {
        "num_gpus": num_gpus,
        "vram_gb": vram_gb,
        "vcpus": vcpus,
        "batch_size": batch_size,
        "eval_batch_size": max(8, batch_size // 2),
        "num_workers": num_workers,
        "eval_workers": max(2, num_workers // 3),
    }


def print_hardware_config(config):
    """Print hardware configuration"""
    print("\n" + "="*80)
    print("⚙️  HARDWARE CONFIGURATION")
    print("="*80)
    print(f"  GPUs: {config['num_gpus']}")
    print(f"  Distributed training: {'Yes' if config['num_gpus'] > 1 else 'No'}")
    if config["vram_gb"]:
        print(f"  VRAM per GPU: {config['vram_gb']:.0f} GB")
    print(f"  vCPUs: {config['vcpus']}")
    print(f"  Batch size (per GPU): {config['batch_size']}")
    print(f"  Eval batch size: {config['eval_batch_size']}")
    print(f"  Data workers: {config['num_workers']}")
    print(f"  Eval workers: {config['eval_workers']}")
    print("="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram", type=int, default=None, help="VRAM in GB")
    parser.add_argument("--vcpus", type=int, default=None, help="Number of vCPUs")
    parser.add_argument("--restore_path", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Pre-trained model path")
    args = parser.parse_args()

    # Hardware config
    hw_config = get_hardware_config(args.vram, args.vcpus)
    print_hardware_config(hw_config)

    # Load phoneme vocabulary (16ms alignments)
    phoneme_map_path = Path("phones_16ms/phoneme_map.json")
    with phoneme_map_path.open("r") as f:
        phoneme_map = json.load(f)

    phoneme_list = sorted(phoneme_map.keys(), key=lambda k: phoneme_map[k])
    num_phonemes = len(phoneme_list)

    print(f"📚 Phonemes: {num_phonemes}")
    print(f"   Sample: {phoneme_list[:10]}\n")

    # Load manifest
    from TTS.tts.datasets.formatters import register_formatter
    register_formatter("native_tts", native_tts_formatter)

    manifest_items = native_tts_formatter(".", "data/manifest_native.jsonl")
    speakers = sorted(set(item["speaker_name"] for item in manifest_items))
    num_speakers = len(speakers)

    # Split train/eval
    train_samples = manifest_items[:int(len(manifest_items) * 0.99)]
    eval_samples = manifest_items[int(len(manifest_items) * 0.99):]

    print(f"📊 Dataset:")
    print(f"   Train: {len(train_samples)}")
    print(f"   Eval: {len(eval_samples)}")
    print(f"   Speakers: {num_speakers}\n")

    # Audio config (16ms frames to match pretrained model)
    audio_config = VitsAudioConfig(
        sample_rate=16000,
        hop_length=256,      # 16ms at 16kHz
        win_length=1024,
        fft_size=1024,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=8000,
    )

    # VITS config - FULLY VALIDATED against v0.27.2
    config = VitsConfig(
        batch_size=hw_config["batch_size"],
        eval_batch_size=hw_config["eval_batch_size"],
        num_loader_workers=hw_config["num_workers"],
        num_eval_loader_workers=hw_config["eval_workers"],
        run_eval=True,
        test_delay_epochs=-1,
        epochs=9999,
        text_cleaner=None,
        use_phonemes=False,
        phoneme_language=None,
        compute_input_seq_cache=False,
        print_step=50,
        print_eval=False,
        mixed_precision=True,
        output_path="runs/native_vits_16k",
        datasets=[
            BaseDatasetConfig(
                formatter="native_tts",
                meta_file_train="data/manifest_native.jsonl",
                path=".",
            )
        ],
        save_step=5000,
        save_n_checkpoints=3,
        save_checkpoints=True,
        lr_gen=0.0002,
        lr_disc=0.0002,
        lr_scheduler_gen="ExponentialLR",
        lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR",
        lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},
        scheduler_after_epoch=False,
        audio=audio_config,
        min_audio_len=0.5 * 16000,
        max_audio_len=10.0 * 16000,
        min_text_len=1,
        max_text_len=500,
        use_speaker_embedding=False,
        use_d_vector_file=True,
        d_vector_dim=192,
        num_speakers=num_speakers,
    )

    # Create characters using BaseVocabulary (for multi-char phoneme tokens)
    from TTS.tts.utils.text.characters import BaseVocabulary
    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    characters = BaseVocabulary(
        vocab=phoneme_list,  # List of phoneme tokens (78 phonemes)
        pad="sil",          # Use silence as padding (index 0)
        blank=None,
        bos=None,
        eos=None,
    )

    print(f"✓ Vocabulary size: {characters.num_chars}\n")

    tokenizer = TTSTokenizer(
        use_phonemes=False,
        text_cleaner=None,
        characters=characters,
    )

    config.characters = characters.to_config()

    # Initialize audio processor and model
    ap = AudioProcessor.init_from_config(config)
    model = Vits(config, ap, tokenizer=tokenizer)

    # Override get_data_loader to use our custom dataset
    def get_data_loader(config, assets, is_eval, samples, verbose, num_gpus, **kwargs):
        dataset = NativeTTSDataset(samples, ap, d_vector_dim=192)

        sampler = None
        shuffle = not is_eval

        if num_gpus > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset)
            shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=True,
            drop_last=not is_eval,
        )
        return loader

    model.get_data_loader = get_data_loader

    # Load pre-trained weights
    if args.pretrained_path:
        model = load_pretrained_weights(model, args.pretrained_path)
    elif args.restore_path is None:
        pretrained_path = download_pretrained_model()
        model = load_pretrained_weights(model, pretrained_path)

    # Trainer args
    trainer_args = TrainerArgs(
        restore_path=args.restore_path,
        skip_train_epoch=False,
        use_accelerate=False,
        use_ddp=hw_config["num_gpus"] > 1,
    )

    # Initialize trainer
    trainer = Trainer(
        trainer_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Print training info
    print("\n" + "="*80)
    print("🚀 STARTING NATIVE TTS TRAINING (Stage 1)")
    print("="*80)
    print(f"  Model: Multi-Speaker VITS")
    print(f"  Phonemes: {num_phonemes}")
    print(f"  Speakers: {num_speakers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Transfer learning: Yes (encoder/decoder from pretrained)")
    print()
    print(f"  📁 Checkpoints: {config.output_path}")
    print(f"  💾 Save frequency: Every {config.save_step} steps")
    print(f"  📊 Print frequency: Every {config.print_step} steps")
    print()
    print(f"  ⏱️  Expected training time:")
    print(f"     • 24h  → 15k-20k steps (basic quality)")
    print(f"     • 48h  → 30k-40k steps (good quality)")
    print(f"     • 72h+ → 50k+ steps (Stage 2 ready)")
    print()
    print(f"  ⚠️  Press Ctrl+C anytime to save checkpoint and exit")
    print("="*80 + "\n")

    # Start training
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⚠️  Training interrupted - saving checkpoint...")
        print("="*80)
        trainer.save_checkpoint()
        sys.exit(0)


if __name__ == "__main__":
    main()
