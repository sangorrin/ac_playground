#!/usr/bin/env python3
"""
NUCLEAR OPTION - Force multi-speaker at the model level
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
from TTS.tts.models.vits import Vits, VitsAudioConfig, VitsArgs
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.utils.speakers import SpeakerManager

class NativeTTSDataset(Dataset):
    def __init__(self, samples, ap):
        self.samples = samples
        self.ap = ap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav = self.ap.load_wav(item["audio_file"])
        phoneme_ids = np.load(item["phoneme_file"]).astype(np.int64)

        return {
            "raw_text": item["text"],
            "token_ids": phoneme_ids,
            "token_len": len(phoneme_ids),
            "wav": wav,
            "wav_len": len(wav),
            "speaker_name": item["speaker_name"],
            "audio_file": item["audio_file"],
            "audio_unique_name": Path(item["audio_file"]).stem,
            "language_name": "native",
        }

def collate_fn(batch):
    max_token_len = max(x["token_len"] for x in batch)
    max_wav_len = max(x["wav_len"] for x in batch)
    B = len(batch)

    token_padded = torch.zeros(B, max_token_len, dtype=torch.long)
    token_lens = torch.LongTensor([x["token_len"] for x in batch])
    token_rel_lens = token_lens.float() / max_token_len
    wav_padded = torch.zeros(B, 1, max_wav_len, dtype=torch.float)
    wav_lens = torch.LongTensor([x["wav_len"] for x in batch])
    wav_rel_lens = wav_lens.float() / max_wav_len

    speaker_names = [x["speaker_name"] for x in batch]
    language_names = [x["language_name"] for x in batch]
    audio_files = [x["audio_file"] for x in batch]
    raw_texts = [x["raw_text"] for x in batch]
    audio_unique_names = [x["audio_unique_name"] for x in batch]

    for i, item in enumerate(batch):
        token_padded[i, :item["token_len"]] = torch.from_numpy(item["token_ids"])
        wav = torch.from_numpy(item["wav"])
        wav_padded[i, 0, :item["wav_len"]] = wav

    return {
        "tokens": token_padded, "token_lens": token_lens, "token_rel_lens": token_rel_lens,
        "waveform": wav_padded, "waveform_lens": wav_lens, "waveform_rel_lens": wav_rel_lens,
        "speaker_names": speaker_names, "language_names": language_names, "audio_files": audio_files,
        "raw_text": raw_texts, "audio_unique_names": audio_unique_names,
    }

def native_tts_formatter(root_path, meta_file, **kwargs):
    items = []
    manifest_path = Path(root_path) / meta_file
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            items.append({
                "text": data["text"],
                "audio_file": str(Path(root_path) / data["audio_file"]),
                "phoneme_file": str(Path(root_path) / data["phoneme_file"]),
                "speaker_name": data["speaker_name"],
            })
    return items

def get_hardware_config(vram_gb=None, vcpus=None):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if vram_gb is None and num_gpus > 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vcpus is None:
        vcpus = os.cpu_count() or 8
    batch_size = max(8, int((vram_gb - 2) * 1.1)) if vram_gb else 16
    batch_size = min(batch_size, 32)
    num_workers = max(4, min(vcpus - 2, 8))
    return {"num_gpus": num_gpus, "batch_size": batch_size, "eval_batch_size": max(8, batch_size // 2), "num_workers": num_workers, "eval_workers": max(2, num_workers // 3)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram", type=int, default=None)
    parser.add_argument("--vcpus", type=int, default=None)
    parser.add_argument("--restore_path", type=str, default=None)
    args = parser.parse_args()

    hw_config = get_hardware_config(args.vram, args.vcpus)

    # Load phonemes
    phoneme_map_path = Path("phones_16ms/phoneme_map.json")
    with phoneme_map_path.open("r") as f:
        phoneme_map = json.load(f)
    phoneme_list = sorted(phoneme_map.keys(), key=lambda k: phoneme_map[k])
    num_phonemes = len(phoneme_list)

    # Load manifest
    from TTS.tts.datasets.formatters import register_formatter
    register_formatter("native_tts", native_tts_formatter)
    manifest_items = native_tts_formatter(".", "data/manifest_native.jsonl")
    speakers = sorted(set(item["speaker_name"] for item in manifest_items))
    num_speakers = len(speakers)
    train_samples = manifest_items[:int(len(manifest_items) * 0.99)]
    eval_samples = manifest_items[int(len(manifest_items) * 0.99):]

    # Audio config
    audio_config = VitsAudioConfig(sample_rate=16000, hop_length=256, win_length=1024, fft_size=1024, num_mels=80, mel_fmin=0, mel_fmax=8000)

    # NUCLEAR OPTION: Force model_args BEFORE creating config
    forced_model_args = VitsArgs(
        num_chars=num_phonemes,
        use_speaker_embedding=True,
        num_speakers=num_speakers,
        speaker_embedding_channels=256,
        out_channels=513, spec_segment_size=32, hidden_channels=192,
        hidden_channels_ffn_text_encoder=768, num_heads_text_encoder=2,
        num_layers_text_encoder=6, kernel_size_text_encoder=3,
        dropout_p_text_encoder=0.1, dropout_p_duration_predictor=0.5,
        kernel_size_posterior_encoder=5, dilation_rate_posterior_encoder=1,
        num_layers_posterior_encoder=16, kernel_size_flow=5, dilation_rate_flow=1,
        num_layers_flow=4, resblock_type_decoder="1",
        resblock_kernel_sizes_decoder=[3, 7, 11],
        resblock_dilation_sizes_decoder=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates_decoder=[8, 8, 2, 2], upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[16, 16, 4, 4],
        periods_multi_period_discriminator=[2, 3, 5, 7, 11],
        use_sdp=True, noise_scale=1.0, inference_noise_scale=0.667,
        length_scale=1, noise_scale_dp=1.0, inference_noise_scale_dp=1.0,
        max_inference_len=None, init_discriminator=True,
        use_spectral_norm_disriminator=False, detach_dp_input=True,
        use_language_embedding=False, embedded_language_dim=4, num_languages=0,
        language_ids_file=None, use_speaker_encoder_as_loss=False,
        speaker_encoder_config_path="", speaker_encoder_model_path="",
        condition_dp_on_speaker=True, freeze_encoder=False, freeze_DP=False,
        freeze_PE=False, freeze_flow_decoder=False, freeze_waveform_decoder=False,
        encoder_sample_rate=None, interpolate_z=True, reinit_DP=False, reinit_text_encoder=False
    )

    # VITS config with FORCED model_args
    config = VitsConfig(
        batch_size=2*hw_config["batch_size"], eval_batch_size=hw_config["eval_batch_size"],
        num_loader_workers=hw_config["num_workers"], num_eval_loader_workers=hw_config["eval_workers"],
        run_eval=True, epochs=9999, print_step=50, mixed_precision=True,
        output_path="runs/native_vits_NUCLEAR",
        datasets=[BaseDatasetConfig(formatter="native_tts", meta_file_train="data/manifest_native.jsonl", path=".")],
        save_step=5000, save_n_checkpoints=3, lr_gen=0.0002, lr_disc=0.0002,
        lr_scheduler_gen="ExponentialLR", lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR", lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},
        audio=audio_config, min_audio_len=0.5 * 16000, max_audio_len=10.0 * 16000, min_text_len=1, max_text_len=500,
        model_args=forced_model_args,  # FORCE IT!
        use_speaker_embedding=True, num_speakers=num_speakers, speaker_embedding_channels=256,
    )

    # Characters
    from TTS.tts.utils.text.characters import BaseVocabulary
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    characters = BaseVocabulary(vocab=phoneme_list, pad="sil", blank=None, bos=None, eos=None)
    tokenizer = TTSTokenizer(use_phonemes=False, text_cleaner=None, characters=characters)
    config.characters = characters.to_config()

    # SpeakerManager
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")

    # Initialize
    ap = AudioProcessor.init_from_config(config)
    model = Vits(config, ap, tokenizer=tokenizer, speaker_manager=speaker_manager)

    print(f"✅ NUCLEAR CHECK:")
    print(f"   use_speaker_embedding: {model.args.use_speaker_embedding}")
    print(f"   num_speakers: {model.args.num_speakers}")
    print(f"   embedded_speaker_dim: {model.embedded_speaker_dim}")

    # Custom data loader
    def get_data_loader(config, assets, is_eval, samples, verbose, num_gpus, **kwargs):
        dataset = NativeTTSDataset(samples, ap)
        return DataLoader(dataset, batch_size=config.eval_batch_size if is_eval else config.batch_size, shuffle=not is_eval, collate_fn=collate_fn, num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers, pin_memory=True, drop_last=not is_eval)
    model.get_data_loader = get_data_loader

    # Trainer
    trainer_args = TrainerArgs(restore_path=args.restore_path, use_ddp=hw_config["num_gpus"] > 1)
    trainer = Trainer(trainer_args, config, config.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples)

    print("🚀 NUCLEAR TRAINING START")
    try:
        trainer.fit()
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        sys.exit(0)

if __name__ == "__main__":
    main()
