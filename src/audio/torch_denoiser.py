"""
src/audio/torch_denoiser.py

Lightweight PyTorch denoiser (U-Net on log-magnitude spectrogram).
Usage:
    model = load_model("models/denoiser.pt", device='cuda', half=False)
    clean_wav = denoise_waveform(model, noisy_wav, sr=16000)
Notes:
 - Model expects mono waveform numpy float32 in [-1,1].
 - If you don't have a checkpoint, load_model(None, ...) will return an uninitialized model you can use for quick tests.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import librosa


# -------------------------
# Model: small 2D U-Net
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (in case of odd sizes)
        if x.size()[-2:] != skip.size()[-2:]:
            x = torch.nn.functional.pad(x, [0, skip.size(-1) - x.size(-1), 0, skip.size(-2) - x.size(-2)])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, depth=4):
        """
        in_ch: input channels (1 = single-channel log-mag)
        base_ch: base channel count
        depth: downsampling depth (4 is reasonable)
        """
        super().__init__()
        self.inc = ConvBlock(in_ch, base_ch)
        self.downs = nn.ModuleList()
        ch = base_ch
        for _ in range(depth - 1):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        self.bottleneck = ConvBlock(ch, ch * 2)

        self.ups = nn.ModuleList()
        for _ in range(depth - 1):
            self.ups.append(Up(ch * 2, ch))
            ch //= 2

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape (B, C=1, F, T)
        skips = []
        x0 = self.inc(x)
        skips.append(x0)
        x = x0
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        x = self.out_conv(x)
        # Output is mask logits; apply sigmoid in inference to get (0,1) mask
        return x


# -------------------------
# STFT helpers
# -------------------------
def _stft_torch(wav: np.ndarray, n_fft=1024, hop_length=256, win_length=None, device='cpu'):
    """
    Returns complex STFT as numpy arrays: (complex_stft, freqs, times)
    complex_stft: np.complex64 array shape (F, T)
    """
    if win_length is None:
        win_length = n_fft
    # use librosa for consistent freq bins (we'll convert back and forth)
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return D  # complex-valued numpy


def _istft_torch(D_complex: np.ndarray, hop_length=256, win_length=None):
    if win_length is None:
        win_length = (D_complex.shape[0] - 1) * 2
    wav = librosa.istft(D_complex, hop_length=hop_length, win_length=win_length)
    return wav


# -------------------------
# Model load / utility
# -------------------------
def load_model(path: Optional[str] = None, device: Optional[str] = None, half: bool = False) -> torch.nn.Module:
    """
    Load a denoiser model.
    - path: path to .pt checkpoint. If checkpoint is a state_dict, it's loaded into the model.
    - device: 'cuda' or 'cpu' or None to auto-detect cuda if available.
    - half: convert to fp16 (only recommended on CUDA).
    Returns the model on device.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetDenoiser(in_ch=1, base_ch=32, depth=4)  # wider base_ch=32 for a bit more capacity
    map_device = torch.device(device)
    model.to(map_device)

    if path:
        ckpt = torch.load(path, map_location=map_device)
        # Accept either full model saved or state_dict
        if isinstance(ckpt, dict) and any(k.startswith('module.') or k.startswith('inc.') or k.startswith('bottleneck') for k in ckpt.keys()):
            # state_dict-like
            try:
                model.load_state_dict(ckpt)
            except RuntimeError:
                # maybe saved with DataParallel
                new = {}
                for k, v in ckpt.items():
                    nk = k.replace('module.', '')
                    new[nk] = v
                model.load_state_dict(new)
        else:
            # assume full model object
            try:
                model = ckpt
                model.to(map_device)
            except Exception:
                # fallback: load into state_dict if possible
                try:
                    model.load_state_dict(ckpt)
                except Exception as e:
                    raise RuntimeError(f"Unknown checkpoint format: {e}")
    model.eval()
    if half and map_device.type == 'cuda':
        model.half()
    return model


def denoise_spectrogram_mask(model: torch.nn.Module,
                             noisy_wav: np.ndarray,
                             sr: int = 16000,
                             n_fft: int = 1024,
                             hop_length: int = 256,
                             win_length: Optional[int] = None,
                             mask_strength: float = 1.0,
                             device: Optional[str] = None) -> np.ndarray:
    """
    Main inference function:
      - compute STFT
      - compute log-mag input
      - run U-Net to predict mask logits
      - apply sigmoid(mask_strength * logits) to get mask in (0,1)
      - apply mask to complex STFT and inverse STFT
    Returns cleaned waveform numpy float32.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    map_device = torch.device(device)

    # mono, numpy float32
    wav = noisy_wav.astype(np.float32)

    # STFT (complex)
    D = _stft_torch(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # shape (F, T) complex
    mag = np.abs(D)  # (F, T)
    # Convert to stabilized log-magnitude
    mag_eps = 1e-8
    log_mag = np.log(mag + mag_eps)

    # Normalize log_mag per sample (simple mean/std)
    mean = log_mag.mean()
    std = log_mag.std() + 1e-9
    norm_log_mag = (log_mag - mean) / std

    # Prepare torch tensor: (B=1, C=1, F, T)
    x = torch.from_numpy(norm_log_mag).unsqueeze(0).unsqueeze(0).to(map_device).float()
    # If model is half, convert x to half
    if next(model.parameters()).dtype == torch.float16:
        x = x.half()

    with torch.no_grad():
        logits = model(x)  # (1,1,F,T)
        logits = logits.squeeze(0).squeeze(0).cpu().numpy()

    # Compute mask
    mask = 1.0 / (1.0 + np.exp(-mask_strength * logits))  # sigmoid

    # Apply mask on magnitude and reconstruct complex STFT
    cleaned_mag = mag * mask
    cleaned_D = cleaned_mag * np.exp(1j * np.angle(D))

    # ISTFT
    wav_out = _istft_torch(cleaned_D, hop_length=hop_length, win_length=win_length)
    # Trim/pad to original length
    if len(wav_out) > len(wav):
        wav_out = wav_out[:len(wav)]
    elif len(wav_out) < len(wav):
        wav_out = np.pad(wav_out, (0, len(wav) - len(wav_out)), mode='constant')

    # Simple normalization
    maxv = max(1e-9, np.max(np.abs(wav_out)))
    wav_out = wav_out / maxv
    return wav_out.astype(np.float32)


def denoise_waveform(model: torch.nn.Module,
                     noisy_wav: np.ndarray,
                     sr: int = 16000,
                     device: Optional[str] = None,
                     n_fft: int = 1024,
                     hop_length: int = 256,
                     win_length: Optional[int] = None,
                     mask_strength: float = 1.0) -> np.ndarray:
    """
    Convenience wrapper expecting numpy waveform. Returns cleaned waveform.
    """
    return denoise_spectrogram_mask(model=model,
                                    noisy_wav=noisy_wav,
                                    sr=sr,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    mask_strength=mask_strength,
                                    device=device)
