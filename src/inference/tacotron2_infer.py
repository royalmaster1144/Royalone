import torch

import numpy as np




def load_model(path: str, device: torch.device, half: bool = False):
    state = torch.load(path, map_location=device)
    model = state
    model.to(device)
    model.eval()
    if half and device.type == "cuda":
        model.half()
    return model




def infer(model, text_seq, speaker_embedding: np.ndarray = None, emotion_vector: np.ndarray = None):
    """Convert text sequence -> mel-spectrogram (n_mel, T) as numpy array.


    `text_seq` is expected to be an integer sequence (list or numpy array). This wrapper
    converts to torch tensors and calls `model.infer` or `model` depending on API.
    """
    import torch as _torch


    # prepare text tensor (B, L)
    if isinstance(text_seq, (list, tuple)):
        text_seq = _torch.LongTensor([text_seq])
    elif isinstance(text_seq, np.ndarray):
        text_seq = _torch.LongTensor(text_seq).unsqueeze(0)


    # speaker embedding
    if speaker_embedding is not None:
        spk = _torch.from_numpy(speaker_embedding).float().unsqueeze(0)
    else:
        spk = None


    if emotion_vector is not None:
        emo = _torch.from_numpy(emotion_vector).float().unsqueeze(0)
    else:
        emo = None


    text_seq = text_seq.to(next(model.parameters()).device)
    if spk is not None:
        spk = spk.to(next(model.parameters()).device)
    if emo is not None:
        emo = emo.to(next(model.parameters()).device)


    with _torch.no_grad():
        # Try the common `infer` signature, otherwise fall back to forward
        try:
            mel = model.infer(text_seq, spk, emo)
        except Exception:
            mel = model(text_seq, spk, emo)
        mel = mel.squeeze(0).cpu().numpy()
        # Ensure shape (n_mel, T)
        return mel.astype(np.float32)