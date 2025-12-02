import numpy as np
from pathlib import Path
import hashlib


def save_embedding(embedding: np.ndarray, save_path: str):
    """
    Save speaker embedding to a .npy file.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    np.save(save_path, embedding)


def load_embedding(load_path: str) -> np.ndarray:
    """
    Load speaker embedding from a .npy file.
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Embedding file not found: {load_path}")
    return np.load(load_path)


# ✅ Optional: Auto-cache embedding using audio file hash
def get_cached_embedding_path(audio_path: str, cache_dir="cache/embeddings"):
    """
    Creates a unique cache file name using audio hash.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(audio_path, "rb") as f:
        audio_hash = hashlib.md5(f.read()).hexdigest()

    return cache_dir / f"{audio_hash}.npy"
