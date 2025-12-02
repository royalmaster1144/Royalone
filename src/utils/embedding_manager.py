import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class EmbeddingManager:
    """
    Manages multiple speaker embeddings saved as .npy files with a small JSON index.

    Directory layout (example):
        cache/embeddings/
            index.json
            <speaker_id>.npy
            <speaker_id2>.npy
    """

    def __init__(self, root_dir: str = "cache/embeddings"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._index = self._load_index()

    # ---------- internal index handling ----------

    def _load_index(self) -> Dict[str, Dict]:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                # corrupted index -> reset
                return {}
        return {}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    # ---------- public API ----------

    def list_speakers(self) -> List[Dict]:
        """
        Returns list of speakers with basic metadata:
        [{ "id": "speaker1", "name": "Rahul Angry", "path": "...", "notes": "optional" }, ...]
        """
        speakers = []
        for spk_id, info in self._index.items():
            SpeakersData = {
                "id": spk_id,
                "name": info.get("name", spk_id),
                "path": str(self.root / f"{spk_id}.npy"),
                "notes": info.get("notes", ""),
            }
            speakers.append(SpeakersData)
        return speakers

    def speaker_exists(self, speaker_id: str) -> bool:
        return speaker_id in self._index and (self.root / f"{speaker_id}.npy").exists()

    def save_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        name: Optional[str] = None,
        notes: str = "",
        overwrite: bool = False,
    ):
        """
        Save a speaker embedding under a given ID.
        speaker_id: internal key (no spaces recommended)
        name: display name (can have spaces)
        """
        speaker_id = speaker_id.strip()
        if not speaker_id:
            raise ValueError("speaker_id cannot be empty")

        fname = self.root / f"{speaker_id}.npy"
        if fname.exists() and not overwrite:
            raise FileExistsError(f"Embedding for speaker_id '{speaker_id}' already exists. Use overwrite=True.")

        embedding = np.asarray(embedding, dtype=np.float32)
        np.save(fname, embedding)

        self._index[speaker_id] = {
            "name": name or speaker_id,
            "notes": notes,
        }
        self._save_index()

    def load_embedding(self, speaker_id: str) -> np.ndarray:
        if speaker_id not in self._index:
            raise KeyError(f"Speaker ID not found in index: {speaker_id}")
        fname = self.root / f"{speaker_id}.npy"
        if not fname.exists():
            raise FileNotFoundError(f"Embedding file missing: {fname}")
        return np.load(fname)

    def delete_speaker(self, speaker_id: str):
        """Remove speaker embedding and metadata."""
        fname = self.root / f"{speaker_id}.npy"
        if fname.exists():
            fname.unlink()
        if speaker_id in self._index:
            del self._index[speaker_id]
            self._save_index()

    def rename_speaker(self, speaker_id: str, new_name: str):
        """Rename display name only (not the ID or file name)."""
        if speaker_id not in self._index:
            raise KeyError(f"Speaker ID not found: {speaker_id}")
        self._index[speaker_id]["name"] = new_name
        self._save_index()
