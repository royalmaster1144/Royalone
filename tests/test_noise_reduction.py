import numpy as np
import soundfile as sf
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio import noise_reduction

def generate_noisy_signal(duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(duration * sr))
    # Clean signal: 440Hz sine wave
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Noise: white noise
    noise = 0.1 * np.random.randn(len(t))
    return (clean + noise).astype(np.float32), clean.astype(np.float32)

def test_spectral_gating():
    print("Testing Spectral Gating...")
    noisy, clean = generate_noisy_signal()
    
    # Run denoise
    denoised = noise_reduction.denoise(noisy, sr=16000, method="spectral")
    
    # Check if noise is reduced (simple energy check)
    noise_energy_before = np.sum((noisy - clean)**2)
    noise_energy_after = np.sum((denoised - clean)**2)
    
    print(f"Noise Energy Before: {noise_energy_before:.4f}")
    print(f"Noise Energy After:  {noise_energy_after:.4f}")
    
    if noise_energy_after < noise_energy_before:
        print("SUCCESS: Noise energy reduced.")
    else:
        print("FAILURE: Noise energy not reduced.")
        
    # Save files for listening
    os.makedirs("tmp_tests", exist_ok=True)
    sf.write("tmp_tests/noisy.wav", noisy, 16000)
    sf.write("tmp_tests/clean.wav", clean, 16000)
    sf.write("tmp_tests/denoised_spectral.wav", denoised, 16000)
    print("Saved test files to tmp_tests/")

def test_simple_subtraction():
    print("\nTesting Simple Subtraction...")
    # This method assumes constant noise profile (like DC or specific pattern)
    # Our white noise is random, so simple subtraction of mean won't do much for variance,
    # but let's test it anyway.
    noisy, clean = generate_noisy_signal()
    
    denoised = noise_reduction.denoise(noisy, sr=16000, method="simple")
    
    noise_energy_before = np.sum((noisy - clean)**2)
    noise_energy_after = np.sum((denoised - clean)**2)
    
    print(f"Noise Energy Before: {noise_energy_before:.4f}")
    print(f"Noise Energy After:  {noise_energy_after:.4f}")
    
    sf.write("tmp_tests/denoised_simple.wav", denoised, 16000)

if __name__ == "__main__":
    try:
        test_spectral_gating()
        test_simple_subtraction()
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure numpy, librosa, and soundfile are installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
