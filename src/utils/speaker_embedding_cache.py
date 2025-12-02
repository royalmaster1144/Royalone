import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np


def _hash_bytes(data: bytes) -> str:
    """Return a stable hash for any audio bytes."""
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str) -> str:
    """Return hash of a file's content (for disk-based caching)."""
    p = Path(path)
    with p.open("rb") as f:
        data = f.read()
    return _hash_bytes(data)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _embedding_path_from_hash(hash_str: str, cache_dir: str) -> Path:
    cache_dir = Path(cache_dir)
    _ensure_dir(cache_dir)
    return cache_dir / f"{hash_str}.npy"


def save_embedding_to_disk(embedding: np.ndarray, hash_str: str, cache_dir: str) -> Path:
    emb = np.asarray(embedding, dtype=np.float32)
    path = _embedding_path_from_hash(hash_str, cache_dir)
    np.save(path, emb)
    return path


def load_embedding_from_disk(hash_str: str, cache_dir: str) -> Optional[np.ndarray]:
    path = _embedding_path_from_hash(hash_str, cache_dir)
    if path.exists():
        return np.load(path)
    return None


# --------- PUBLIC API ---------

@lru_cache(maxsize=256)
def get_or_compute_embedding_from_bytes(
    audio_bytes: bytes,
    sr: int,
    spk_model,
    infer_fn,
    cache_dir: str = "cache/embeddings",
) -> np.ndarray:
    """
    Use this in Streamlit (uploaded files):
    - audio_bytes: raw bytes from uploaded file
    - sr: sample rate after your load_wav
    - spk_model: loaded speaker encoder model
    - infer_fn: function like infer_spk(model, wav, sample_rate)
    """
    import io
    import soundfile as sf

    # 1) compute stable hash
    h = _hash_bytes(audio_bytes)

    # 2) disk cache
    emb = load_embedding_from_disk(h, cache_dir)
    if emb is not None:
        return emb

    # 3) decode audio, compute embedding
    data, file_sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    # resample if needed
    if file_sr != sr:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=file_sr, target_sr=sr)

    emb = infer_fn(spk_model, data.astype(np.float32), sample_rate=sr)

    # 4) save to disk + return
    save_embedding_to_disk(emb, h, cache_dir)
    return emb


@lru_cache(maxsize=256)
def get_or_compute_embedding_from_file(
    audio_path: str,
    sr: int,
    spk_model,
    infer_fn,
    cache_dir: str = "cache/embeddings",
) -> np.ndarray:
    """
    Use this if you have an actual file path on disk:
    - audio_path: path to wav/mp3 file
    - sr: target sample rate
    - spk_model: loaded speaker encoder model
    - infer_fn: function like infer_spk(model, wav, sample_rate)
    """
    from src.audio.audio_utils import load_wav  # adjust import if needed

    # 1) compute file hash
    h = _hash_file(audio_path)

    # 2) disk cache
    emb = load_embedding_from_disk(h, cache_dir)
    if emb is not None:
        return emb

    # 3) load audio, compute embedding
    wav, _sr = load_wav(audio_path, sr=sr)
    emb = infer_fn(spk_model, wav, sample_rate=sr)

    # 4) save to disk + return
    save_embedding_to_disk(emb, h, cache_dir)
    return emb
