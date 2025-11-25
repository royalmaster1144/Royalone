import librosa
import numpy as np
import soundfile as sf
from typing import Tuple

def load_wav(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to sr. Returns float32 mono audio in [-1,1]."""
    wav, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        wav = librosa.resample(y=wav, orig_sr=orig_sr, target_sr=sr)
    wav = wav.astype(np.float32)
    # normalize safely
    maxv = max(1e-9, float(np.max(np.abs(wav))))
    wav = wav / maxv
    return wav, sr

def save_wav(path: str, wav: np.ndarray, sr: int = 16000):
    sf.write(path, wav.astype(np.float32), sr)

def normalize_audio(wav: np.ndarray) -> np.ndarray:
    maxv = max(1e-9, float(np.max(np.abs(wav))))
    return wav / maxv

def extract_mel(wav: np.ndarray, sr: int = 16000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80):
    """Return melspectrogram (float32) compatible with many TTS pipelines.

    Output shape: (n_mels, T)
    """
    S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0)
    # Apply log to match many models' dB-like inputs, add small eps
    S = np.log(np.maximum(S, 1e-10))
    return S.astype(np.float32)