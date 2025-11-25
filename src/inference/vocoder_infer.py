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




def infer(model, mel: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Convert mel (n_mel, T) -> waveform numpy float32 in [-1,1]."""
    import torch as _torch


    if isinstance(mel, np.ndarray):
        mel_t = _torch.from_numpy(mel).float().unsqueeze(0)
    else:
        mel_t = mel


    mel_t = mel_t.to(next(model.parameters()).device)


    with _torch.no_grad():
        try:
            wav = model.infer(mel_t)
        except Exception:
            wav = model(mel_t)
        wav = wav.squeeze(0).cpu().numpy()
        # clip and normalize a bit
        maxv = max(1e-9, float(np.max(np.abs(wav))))
        wav = wav / maxv
    return wav.astype(np.float32)