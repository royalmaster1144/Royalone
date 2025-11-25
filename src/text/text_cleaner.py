import re
from typing import List


try:
    from phonemizer import phonemize
    PHONEMIZER_AVAILABLE = True
except Exception:
    PHONEMIZER_AVAILABLE = False




def clean_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t




def text_to_sequence(text: str, phonemize_out: bool = False) -> List[int]:
    """Very small text -> numeric sequence converter.


    By default returns simple unicode codepoints (not recommended for production).
    If `phonemize_out` is True and the phonemizer is available, return phoneme string tokens
    separated by spaces.
    """
    t = clean_text(text)
    if phonemize_out and PHONEMIZER_AVAILABLE:
        phones = phonemize(t, language='en-us', backend='espeak', strip=True)
        # return as list of ord values per character of the phoneme string
        return [ord(c) for c in phones]
    # fallback: unicode codepoints
    return [ord(c) for c in t]