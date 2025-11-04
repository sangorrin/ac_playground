#!/usr/bin/env python3
"""
Native TTS Inference Script - TEXT-BASED with MAS alignment
Test your trained multi-speaker VITS model using text input (like training)
Uses SPEAKER EMBEDDINGS and MAS alignment (same as training)
"""

import argparse
import json
import os
import sys
import csv
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.characters import BaseVocabulary
from TTS.tts.utils.text.tokenizer import TTSTokenizer

def load_ljs_text(meta_csv: Path) -> dict:
    """Load LJS metadata.csv (pipe-delimited, no quoting) - from ac_playground repo"""
    meta = {}
    with meta_csv.open("r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if not row or len(row) < 2:
                continue
            utt_id = row[0].strip()
            # Prefer normalized text (row[2]), fallback to raw (row[1])
            text = (row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip())
            if utt_id and text:
                meta[utt_id] = text
    return meta

class NativeTTSInference:
    def __init__(self, model_path, config_path, metadata_path=None):
        print(f"🚀 Loading model from: {model_path}")
        print(f"📋 Loading config from: {config_path}")

        # Force CPU to avoid interfering with training
        self.device = torch.device("cpu")
        print(f"💻 Using CPU inference (training-safe)")

        # Load config and reconstruct properly
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Create VitsConfig and set basic attributes
        config = VitsConfig()
        for key, value in config_dict.items():
            if hasattr(config, key) and key != 'model_args':
                setattr(config, key, value)

        # Reconstruct model_args properly
        if 'model_args' in config_dict and config_dict['model_args']:
            model_args_dict = config_dict['model_args']
            model_args = VitsArgs()
            for key, value in model_args_dict.items():
                if hasattr(model_args, key):
                    setattr(model_args, key, value)
            config.model_args = model_args

        # Load LJSpeech metadata (REQUIRED for text input)
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = load_ljs_text(Path(metadata_path))
            print(f"✅ Loaded LJSpeech metadata: {len(self.metadata)} entries")
        else:
            print(f"⚠️  Warning: Metadata required for text-based inference!")

        # Load phoneme vocabulary exactly like training script
        phoneme_map_path = Path("phones_20ms/phoneme_map.json")
        if not phoneme_map_path.exists():
            print(f"❌ Phoneme map not found: {phoneme_map_path}")
            sys.exit(1)

        with phoneme_map_path.open("r") as f:
            self.phoneme_map = json.load(f)
        phoneme_list = sorted(self.phoneme_map.keys(), key=lambda k: self.phoneme_map[k])
        print(f"📚 Loaded {len(phoneme_list)} phoneme symbols (for vocabulary only)")

        # Characters and tokenizer exactly like training script
        characters = BaseVocabulary(
            vocab=phoneme_list, pad="sil", blank=None, bos=None, eos=None
        )
        self.tokenizer = TTSTokenizer(
            use_phonemes=False, text_cleaner=None, characters=characters
        )

        # Set characters in config exactly like training script
        config.characters = characters.to_config()

        # Ensure model_args has the right num_chars
        if config.model_args:
            config.model_args.num_chars = len(phoneme_list)

        # Audio processor
        self.ap = AudioProcessor.init_from_config(config)

        # Speaker manager for SPEAKER EMBEDDINGS (not d-vectors)
        self.speaker_manager = SpeakerManager()
        speakers_file = Path(model_path).parent / "speakers.pth"
        if speakers_file.exists():
            # For speaker embeddings, we load IDs not embeddings
            self.speaker_manager.load_ids_from_file(str(speakers_file))
            print(f"✅ Loaded {self.speaker_manager.num_speakers} speaker IDs from: {speakers_file}")
        else:
            print(f"⚠️  Warning: speakers.pth not found at {speakers_file}")

        # Initialize model exactly like training script
        self.model = Vits(config, self.ap, tokenizer=self.tokenizer, speaker_manager=self.speaker_manager)

        print(f"🔍 Model architecture:")
        print(f"   use_speaker_embedding: {self.model.args.use_speaker_embedding}")
        print(f"   num_speakers: {self.model.args.num_speakers}")
        print(f"   embedded_speaker_dim: {self.model.embedded_speaker_dim}")
        print(f"   num_chars: {self.model.args.num_chars}")
        print(f"   Model type: Text-based VITS with MAS alignment + Speaker Embeddings")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint

        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()

        # Keep on CPU
        self.model = self.model.to(self.device)

        print(f"✅ Text-based VITS model loaded successfully")
        print(f"🎤 Available speakers: {len(self.speaker_manager.speaker_names)}")

    def list_speakers(self):
        """List all available speakers"""
        print(f"\n📋 Available speakers ({len(self.speaker_manager.speaker_names)}):")
        for i, speaker in enumerate(sorted(self.speaker_manager.speaker_names)):
            print(f"  {i+1:3d}: {speaker}")

    def get_ljspeech_text(self, sentence_id):
        """Get text for LJSpeech sentence ID"""
        if not self.metadata:
            return None

        if sentence_id not in self.metadata:
            return None

        return self.metadata[sentence_id]

    def synthesize(self, sentence_id, speaker_name, output_path):
        """Synthesize speech from LJSpeech text and speaker using MAS alignment"""
        print(f"\n🎯 Synthesizing (Text-based with MAS):")
        print(f"   Sentence: {sentence_id}")
        print(f"   Speaker: {speaker_name}")

        # Get text (REQUIRED for text-based model)
        text = self.get_ljspeech_text(sentence_id)
        if not text:
            print(f"❌ Could not find text for {sentence_id}!")
            print(f"   Text-based model requires LJSpeech metadata")
            return None

        print(f"   Text: '{text}'")

        # Check if speaker exists
        if speaker_name not in self.speaker_manager.speaker_names:
            print(f"❌ Speaker '{speaker_name}' not found!")
            self.list_speakers()
            return None

        # Convert text to token IDs using tokenizer (same as training)
        token_ids = self.tokenizer.text_to_ids(text)
        token_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
        print(f"📝 Converted text to {len(token_ids)} tokens using tokenizer")

        # Get speaker ID (for speaker embeddings)
        speaker_id = self.speaker_manager.name_to_id[speaker_name]
        print(f"   Speaker ID: {speaker_id}")

        with torch.no_grad():
            # Prepare inputs for SPEAKER EMBEDDING model with TEXT input
            inputs = {
                "x": token_tensor,
                "x_lengths": torch.LongTensor([token_tensor.shape[1]]).to(self.device),
                "speaker_ids": torch.LongTensor([speaker_id]).to(self.device),
            }

            print(f"🔊 Generating audio from text (MAS alignment, CPU inference)...")
            print(f"   Input shape: {token_tensor.shape}")

            # Generate audio using VITS inference
            outputs = self.model.inference(**inputs)

            # Extract waveform
            if isinstance(outputs, dict) and "wav" in outputs:
                wav = outputs["wav"]
            elif isinstance(outputs, dict) and "model_outputs" in outputs:
                wav = outputs["model_outputs"]
            else:
                wav = outputs

            # Convert to numpy
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy().squeeze()

        # Save audio
        sf.write(output_path, wav, self.ap.sample_rate)
        print(f"✅ Audio saved: {output_path}")
        print(f"   Duration: {len(wav) / self.ap.sample_rate:.2f}s")
        print(f"   Sample rate: {self.ap.sample_rate}Hz")
        print(f"   Compare with FreeVC: {sentence_id}_{speaker_name}.wav")

        return wav, output_path, text


def main():
    parser = argparse.ArgumentParser(description="Native TTS Inference - Text-based with MAS alignment")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint (required)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config.json (required)")
    parser.add_argument("--metadata", type=str, default="metadata.csv",
                       help="Path to LJSpeech metadata.csv (required for text input)")
    parser.add_argument("--ljspeech_sentence", type=str, default="LJ001-0035",
                       help="LJSpeech sentence ID (e.g., LJ001-0035)")
    parser.add_argument("--speaker", type=str, default="p323",
                       help="Speaker ID (e.g., p225, p323, etc.)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output wav file path (default: native_tts_{sentence}_{speaker}.wav)")
    parser.add_argument("--list-speakers", action="store_true",
                       help="List all available speakers")

    args = parser.parse_args()

    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = f"native_tts_{args.ljspeech_sentence}_{args.speaker}.wav"

    # Initialize inference engine
    try:
        tts = NativeTTSInference(args.model, args.config, args.metadata)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # List speakers if requested
    if args.list_speakers:
        tts.list_speakers()
        return

    # Single synthesis
    try:
        wav, output_path, text = tts.synthesize(args.ljspeech_sentence, args.speaker, args.output)

        if wav is not None:
            print(f"\n🎉 Text-based synthesis completed!")
            print(f"   Input: {args.ljspeech_sentence} (text) -> {args.speaker}")
            print(f"   Text: '{text}'")
            print(f"   Output: {output_path}")
            print(f"   Play with: aplay {output_path}")
            print(f"   Compare with FreeVC: {args.ljspeech_sentence}_{args.speaker}.wav")

    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
