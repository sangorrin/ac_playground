import os
import glob
import argparse
import numpy as np
import torch
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, NativeTTSAudioConfig, VitsArgs
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))


def get_hardware_config(vram_gb=None, vcpus=None):
    """Calculate optimal batch size and workers based on hardware specs."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if vram_gb is None and num_gpus > 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vcpus is None:
        vcpus = os.cpu_count() or 8
    batch_size = 2*max(8, int((vram_gb - 2) * 1.1)) if vram_gb else 32
    batch_size = min(batch_size, 32)
    num_workers = max(4, min(vcpus - 2, 8))
    return {
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "eval_batch_size": max(8, batch_size // 2),
        "num_workers": num_workers,
        "eval_workers": max(2, num_workers // 3)
    }


def calculate_num_chars(mfa_dir):
    """Calculate num_chars from MFA alignment files by finding max phoneme ID."""
    max_phoneme_id = 0
    npy_files = glob.glob(os.path.join(mfa_dir, "*.npy"))

    if not npy_files:
        print(f"Warning: No MFA files found in {mfa_dir}, using default num_chars=256")
        return 256

    # Sample some files to find max phoneme ID
    sample_size = min(100, len(npy_files))
    for npy_file in npy_files[:sample_size]:
        try:
            phonemes = np.load(npy_file)
            max_phoneme_id = max(max_phoneme_id, int(phonemes.max()))
        except Exception as e:
            print(f"Warning: Could not load {npy_file}: {e}")
            continue

    # Add buffer for safety (pad token, etc.)
    num_chars = max_phoneme_id + 10
    print(f"Calculated num_chars={num_chars} from MFA files (max_phoneme_id={max_phoneme_id})")
    return num_chars


def native_tts_formatter(root_path, meta_file=None, **kwargs):
    """
    Custom formatter for Native TTS augmented dataset.

    File naming convention:
    - Audio: LJ001-0001_p225.wav (16kHz)
    - MFA alignment: LJ001-0001_p225.npy (phoneme IDs at 20ms frames)
    - F0: LJ001-0001_p225.npy (F0 values at 20ms frames)
    - Speaker embedding: p225.npy (one per VCTK speaker)

    Directory structure:
    root_path/
        wavs_16khz/LJ001-0001_p225.wav
        mfa_alignments/LJ001-0001_p225.npy
        f0_features/LJ001-0001_p225.npy
        speaker_embeddings/p225.npy
    """
    items = []

    wav_dir = os.path.join(root_path, "wavs_16khz")
    mfa_dir = os.path.join(root_path, "mfa_alignments")
    f0_dir = os.path.join(root_path, "f0_features")
    spk_emb_dir = os.path.join(root_path, "speaker_embeddings")

    # Get all audio files: LJ001-0001_p225.wav
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))

    for wav_file in wav_files:
        basename = Path(wav_file).stem  # e.g., "LJ001-0001_p225"

        # Extract speaker ID (e.g., "p225" from "LJ001-0001_p225")
        speaker_id = basename.split("_")[-1]  # Last part after underscore

        # Corresponding files
        mfa_file = os.path.join(mfa_dir, f"{basename}.npy")
        f0_file = os.path.join(f0_dir, f"{basename}.npy")
        spk_emb_file = os.path.join(spk_emb_dir, f"{speaker_id}.npy")

        # Verify all required files exist
        if not os.path.exists(mfa_file):
            print(f"Warning: MFA file not found: {mfa_file}")
            continue
        if not os.path.exists(f0_file):
            print(f"Warning: F0 file not found: {f0_file}")
            continue
        if not os.path.exists(spk_emb_file):
            print(f"Warning: Speaker embedding not found: {spk_emb_file}")
            continue

        items.append({
            "audio_file": wav_file,
            "mfa_file": mfa_file,
            "f0_file": f0_file,
            "speaker_emb_file": spk_emb_file,
            "speaker_name": speaker_id,
            "text": "",  # No text needed - we use MFA phonemes directly
            "audio_unique_name": basename,
            "language": "en"
        })

    return items


def main():
    """
    Train Native TTS model on augmented LJSpeech dataset (FreeVC + VCTK speakers).

    Dataset: ~65k utterances
    - LJSpeech text (13k) x VCTK native English speakers (~5 speakers) = ~65k

    File naming:
    - Audio: LJ001-0001_p225.wav (16kHz, downsampled by 320)
    - MFA: LJ001-0001_p225.npy (aligned phoneme IDs at 20ms frames)
    - F0: LJ001-0001_p225.npy (YAAPTS F0 at 20ms frames)
    - Speaker: p225.npy (ECAPA embedding per VCTK speaker)

    All artifacts precomputed using scripts from sangorrin/ac_playground
    """
    parser = argparse.ArgumentParser(description="Train Native TTS model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/augmented_dataset",
        help="Path to augmented dataset root directory (default: /workspace/augmented_dataset)"
    )
    parser.add_argument(
        "--vram",
        type=float,
        default=None,
        help="GPU VRAM in GB (auto-detected if not provided)"
    )
    parser.add_argument(
        "--vcpus",
        type=int,
        default=None,
        help="Number of CPU cores (auto-detected if not provided)"
    )
    args = parser.parse_args()

    # Get hardware configuration
    hw_config = get_hardware_config(vram_gb=args.vram, vcpus=args.vcpus)
    print(f"Hardware config: {hw_config}")

    # Calculate num_chars from MFA files
    mfa_dir = os.path.join(args.data_path, "mfa_alignments")
    num_chars = calculate_num_chars(mfa_dir)

    # Register custom formatter before loading samples
    from TTS.tts.datasets import FORMATTERS
    FORMATTERS["native_tts_augmented"] = native_tts_formatter

    # Point to your augmented dataset root directory
    dataset_config = BaseDatasetConfig(
        formatter="native_tts_augmented",
        meta_file_train=None,
        path=args.data_path,
    )

    # Native TTS uses 16kHz with 20ms hop (320 samples)
    audio_config = NativeTTSAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=320,  # 20ms at 16kHz
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    # Configure model for Native TTS
    model_args = VitsArgs(
        use_native_tts=True,
        f0_embedding_dim=64,
        use_mfa_alignments=True,
        num_chars=num_chars,
        # Disable duration predictor and MAS (not used in Native TTS)
        use_sdp=False,
        # Multi-speaker with precomputed ECAPA embeddings
        use_speaker_embedding=False,
        use_d_vector_file=False,
        d_vector_dim=192,  # ECAPA embedding dimension (adjust if different)
        # All processing at 16kHz
        encoder_sample_rate=16000,
        interpolate_z=False,
        # Upsampling rates for 320 hop_length: 8*8*5*1 = 320
        upsample_rates_decoder=[8, 8, 5, 1],
        upsample_kernel_sizes_decoder=[16, 16, 10, 2],
    )

    config = VitsConfig(
        model_args=model_args,
        audio=audio_config,
        run_name="native_tts_ljspeech_freevc_vctk",
        batch_size=hw_config["batch_size"],
        eval_batch_size=hw_config["eval_batch_size"],
        batch_group_size=5,
        num_loader_workers=hw_config["num_workers"],
        num_eval_loader_workers=hw_config["eval_workers"],
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        # No text processing - we use MFA phonemes directly
        text_cleaner=None,
        use_phonemes=False,
        phoneme_language=None,
        compute_input_seq_cache=False,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=None,  # No test sentences - would need MFA/F0 at inference
    )

    # INITIALIZE THE AUDIO PROCESSOR
    ap = AudioProcessor.init_from_config(config)

    # Create a minimal tokenizer for Native TTS
    # We don't use it for text processing, only for pad_id
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} eval samples")
    if len(train_samples) > 0:
        print(f"Sample: {train_samples[0]}")

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
