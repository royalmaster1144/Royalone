from functools import lru_cache
import numpy as np
from pathlib import Path




@lru_cache(maxsize=256)
def cache_embedding_from_path(path: str):
"""Placeholder. Real embedding pipeline should call speaker encoder and then store/retrieve.


This function demonstrates caching by filename. Teams should replace with hash-based cache if
multiple files with identical content are possible.
"""
p = Path(path)
if not p.exists():
raise FileNotFoundError(path)
# For now return None to indicate not cached; pipeline will compute and can use lru_cache
return None