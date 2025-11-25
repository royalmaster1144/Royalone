import torch

import numpy as np




def load_model(path: str, device: torch.device, half: bool = False):
    """Load speaker encoder model from a .pt file.


    The expected API on the real model (Team1) is:
    - model.encode_batch(waveform_tensor) -> embedding_tensor (B, 256)


    If your model uses a different API, adjust `infer` accordingly.
    """
    state = torch.load(path, map_location=device)
    # If teams return a raw state dict, they should provide a small wrapper or script to load.
    # We assume they saved the full model object; otherwise the user will replace this.
    model = state
    model.to(device)
    model.eval()
    if half and device.type == "cuda":
        model.half()
    return model




def infer(model, wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Return a 1D numpy embedding (256,).


    This function assumes `model.encode_batch` accepts a torch.FloatTensor of shape (1, N)
    and returns (1, 256). Replace with your model API if different.
    """
    import torch as _torch


    if isinstance(wav, np.ndarray):
        wav_t = _torch.from_numpy(wav).float().unsqueeze(0) # (1, samples)
    else:
        wav_t = wav


    wav_t = wav_t.to(next(model.parameters()).device)
    with _torch.no_grad():
        # many speaker encoders expect normalized waveform in [-1,1]
        try:
            emb = model.encode_batch(wav_t)
        except Exception:
            # fallback if model has encode or forward
            emb = model(wav_t)
        emb = emb.detach().cpu().numpy().squeeze()
    return emb.astype(np.float32)