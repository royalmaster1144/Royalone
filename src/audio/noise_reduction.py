import numpy as np

def _stft(x, n_fft, hop_length):
    window = np.hanning(n_fft)
    n_frames = (len(x) - n_fft) // hop_length + 1
    if n_frames <= 0:
        return np.array([])
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = x[start:start + n_fft] * window
        stft_matrix[:, i] = np.fft.rfft(frame, n=n_fft)
    return stft_matrix

def _istft(stft_matrix, hop_length):
    n_fft = 2 * (stft_matrix.shape[0] - 1)
    n_frames = stft_matrix.shape[1]
    x_len = n_fft + (n_frames - 1) * hop_length
    x = np.zeros(x_len)
    window_sum = np.zeros(x_len)
    window = np.hanning(n_fft)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        x[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
        
    mask = window_sum > 1e-10
    x[mask] /= window_sum[mask]
    return x

def remove_noise_spectral_gate(
        wav,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        noise_reduction_factor=1.5
    ):
    """
    Noise removal using spectral gating (Pure Numpy).
    Returns a cleaned waveform.
    """
    # STFT
    stft = _stft(wav, n_fft=n_fft, hop_length=hop_length)
    if stft.size == 0:
        return wav
        
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise from the first 0.5 sec
    noise_frames = int((0.5 * sr) // hop_length)
    if noise_frames > magnitude.shape[1]:
        noise_frames = magnitude.shape[1]
        
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Create a threshold gate
    threshold = noise_mag * noise_reduction_factor

    # Apply mask
    mask = magnitude > threshold
    cleaned_mag = magnitude * mask

    # Reconstruct
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    cleaned_wav = _istft(cleaned_stft, hop_length=hop_length)

    return cleaned_wav.astype(np.float32)

def remove_noise_profile_subtraction(wav, sr=16000, noise_seconds=0.5):
    """
    Simple noise-profile subtraction.
    Use when the noise is constant (fan, hum).
    """
    noise_len = int(noise_seconds * sr)
    if noise_len > len(wav):
        noise_len = len(wav)
        
    noise_profile = wav[:noise_len]

    # average noise amplitude
    noise_mean = np.mean(noise_profile)

    # subtract noise
    cleaned = wav - noise_mean

    # normalize
    maxv = max(1e-9, np.max(np.abs(cleaned)))
    cleaned = cleaned / maxv

    return cleaned.astype(np.float32)

# Unified function for your main pipeline
def denoise(wav, sr=16000, method="spectral"):
    """
    Wrapper method:
    method = 'spectral'  -> uses spectral gating
    method = 'simple'    -> uses profile subtraction
    """
    if method == "spectral":
        return remove_noise_spectral_gate(wav, sr=sr)
    elif method == "simple":
        return remove_noise_profile_subtraction(wav, sr=sr)
    else:
        raise ValueError("Unknown method. Use 'spectral' or 'simple'.")
