"""
preprocess.py
--------------
Audio preprocessing utilities for the Speech Emotion Detection project.

Responsibilities:
    1. Load an audio file from disk.
    2. Convert it to mono.
    3. Resample it to 16 kHz (the sampling rate Wav2Vec2 was pretrained on).
    4. Normalize the waveform amplitude.
    5. Trim leading/trailing silence.
    6. Extract model-ready features using the Wav2Vec2 processor.

This module is shared by both train.py (offline, batch preprocessing of the
TESS dataset) and predict.py / app.py (online, single-file preprocessing for
inference).
"""

import librosa
import numpy as np
from transformers import Wav2Vec2Processor

# Wav2Vec2 was pretrained on 16 kHz mono audio -- every clip must match this.
TARGET_SAMPLE_RATE = 16000

# Base checkpoint used both for fine-tuning and for building the feature
# extractor/processor used at inference time.
BASE_MODEL_CHECKPOINT = "facebook/wav2vec2-base"


def load_audio(file_path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file, convert it to mono, and resample to `sample_rate`.

    librosa.load() already handles mono conversion (mono=True by default)
    and resampling, so we lean on it directly instead of re-implementing
    that logic.

    Args:
        file_path: Path to the audio file (wav/mp3/flac/etc).
        sample_rate: Target sampling rate in Hz.

    Returns:
        1-D numpy float32 array containing the waveform.
    """
    waveform, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def trim_silence(waveform: np.ndarray, top_db: int = 25) -> np.ndarray:
    """
    Trim leading/trailing silence from a waveform.

    Args:
        waveform: 1-D audio signal.
        top_db: Silence threshold in decibels below peak. Lower values trim
                more aggressively.

    Returns:
        Trimmed waveform. Falls back to the original signal if trimming
        would remove the entire clip (e.g. very quiet recordings).
    """
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    if trimmed.size == 0:
        return waveform
    return trimmed


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """
    Peak-normalize a waveform to the range [-1, 1].

    Args:
        waveform: 1-D audio signal.

    Returns:
        Normalized waveform. Silent input is returned unchanged to avoid
        division by zero.
    """
    peak = np.max(np.abs(waveform))
    if peak < 1e-8:
        return waveform
    return waveform / peak


def preprocess_audio(file_path: str, trim: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline: load -> mono -> resample -> trim -> normalize.

    Args:
        file_path: Path to the raw audio file.
        trim: Whether to trim silence (kept optional since some very short
              TESS clips can become empty after aggressive trimming).

    Returns:
        Clean, normalized, 16 kHz mono waveform ready for feature extraction.
    """
    waveform = load_audio(file_path)
    if trim:
        waveform = trim_silence(waveform)
    waveform = normalize_audio(waveform)
    return waveform


def get_processor(checkpoint: str = BASE_MODEL_CHECKPOINT) -> Wav2Vec2Processor:
    """
    Load (and cache via HF's own caching) the Wav2Vec2 processor, which
    wraps the feature extractor used to turn raw waveforms into the
    normalized input tensors the model expects.
    """
    return Wav2Vec2Processor.from_pretrained(checkpoint)


def extract_features(waveform: np.ndarray, processor: Wav2Vec2Processor,
                      max_duration_seconds: float = 6.0) -> np.ndarray:
    """
    Convert a raw waveform into the input array expected by Wav2Vec2.

    Args:
        waveform: Preprocessed 1-D audio signal at 16 kHz.
        processor: A Wav2Vec2Processor instance.
        max_duration_seconds: Clips are padded/truncated to this length so
                               batches have a uniform shape during training.

    Returns:
        1-D numpy array of processed input values (input_values[0]).
    """
    max_length = int(TARGET_SAMPLE_RATE * max_duration_seconds)
    inputs = processor(
        waveform,
        sampling_rate=TARGET_SAMPLE_RATE,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return np.array(inputs["input_values"][0], dtype=np.float32)
