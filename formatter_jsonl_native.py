import json
from pathlib import Path
from TTS.tts.datasets.formatters import register_formatter

# ---- helpers ---------------------------------------------------------------

def _load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON at {p}:{ln}: {e}")

def _to_sample(obj, root: Path):
    """
    Normalize a manifest row into a dict with required keys:
      audio_file, text, speaker_name, d_vector_file
    Accepts:
      - dict  : {"audio_file":..., "text":..., "speaker_name":..., "d_vector_file":...}
      - list/tuple: [audio_file, text, speaker_name, d_vector_file]
      - string:  "audio_file|text|speaker_name|d_vector_file"
    """
    # dict path (preferred)
    if isinstance(obj, dict):
        audio = obj.get("audio_file") or obj.get("wav") or obj.get("audio")
        text  = obj.get("text") or obj.get("sentence") or obj.get("transcript")
        spk   = obj.get("speaker_name") or obj.get("speaker") or obj.get("spk") or ""
        dvec  = obj.get("d_vector_file") or obj.get("dvec") or obj.get("speaker_embedding")
    # list/tuple path
    elif isinstance(obj, (list, tuple)):
        if len(obj) < 4:
            raise ValueError(f"List/tuple row must have 4 fields, got {len(obj)}")
        audio, text, spk, dvec = obj[:4]
    # string path (pipe-delimited)
    elif isinstance(obj, str):
        parts = obj.split("|")
        if len(parts) < 4:
            raise ValueError(f"String row must be 'audio|text|speaker|dvec', got: {obj}")
        audio, text, spk, dvec = parts[:4]
    else:
        raise TypeError(f"Unsupported row type: {type(obj)}")

    if not audio or not text:
        raise ValueError("Row missing required 'audio_file' or 'text'.")

    audio = Path(audio)
    if not audio.is_absolute():
        audio = (root / audio).resolve()

    dvec = Path(dvec)
    if not dvec.is_absolute():
        dvec = (root / dvec).resolve()

    return {
        "audio_file": str(audio),
        "text": str(text),
        "speaker_name": str(spk or ""),
        "d_vector_file": str(dvec),
    }

# ---- formatter -------------------------------------------------------------

def jsonl_native(root_path: str, meta_file_train: str = None, meta_file_val: str = None, **kwargs):
    """
    Returns (train_samples, val_samples) as lists of dicts with:
      audio_file, text, speaker_name, d_vector_file
    """
    root = Path(root_path or ".")

    def _build(meta_file):
        if not meta_file:
            return []
        mf = (root / meta_file).resolve()
        rows = []
        for obj in _load_jsonl(mf):
            rows.append(_to_sample(obj, root))
        return rows

    train = _build(meta_file_train)
    val = _build(meta_file_val)
    return train, val

# register with the current TTS version (non-decorator API)
register_formatter("jsonl_native", jsonl_native)
